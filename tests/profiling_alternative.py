#!/usr/bin/env python3
"""
CUDA MatMul 性能分析工具 - 权限友好版本
当 Nsight Compute 权限不足时，提供替代的分析方法
"""

import torch
import time
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

def check_ncu_permissions() -> bool:
    """检查 NCU 权限"""
    try:
        # 创建一个简单的测试脚本
        test_script = Path("temp_permission_test.py")
        test_script.write_text('''
import torch
if torch.cuda.is_available():
    a = torch.randn(2, 2, device='cuda')
    print("Simple CUDA test completed")
''')
        
        # 尝试运行 NCU
        result = subprocess.run([
            'ncu', '--target-processes', 'all', '--launch-count', '1',
            '--export', 'temp_test.ncu-rep', '--force-overwrite',
            'python', str(test_script)
        ], capture_output=True, text=True, timeout=30)
        
        # 清理
        test_script.unlink(missing_ok=True)
        Path('temp_test.ncu-rep').unlink(missing_ok=True)
        
        if "ERR_NVGPUCTRPERM" in result.stderr:
            return False
        elif result.returncode == 0:
            return True
        else:
            return False
            
    except Exception:
        return False

def provide_permission_guidance():
    """提供权限设置指导"""
    print("\n" + "="*60)
    print("Nsight Compute 权限设置指导")
    print("="*60)
    print("\nNsight Compute 需要特殊权限来访问 GPU 性能计数器。")
    print("请尝试以下解决方案：\n")
    
    print("方案 1: 临时使用 sudo 运行")
    print("  sudo python ncu_profiler.py --sizes 512")
    print()
    
    print("方案 2: 设置驱动参数 (需要重启)")
    print("  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf")
    print("  sudo update-initramfs -u")
    print("  sudo reboot")
    print()
    
    print("方案 3: 使用 Docker 容器")
    print("  nvidia-docker run --privileged -it nvidia/cuda:12.0-devel")
    print()
    
    print("方案 4: 使用替代的性能分析方法")
    print("  - PyTorch Profiler (无权限要求)")
    print("  - 手动时间测量和 GFLOPS 计算")
    print("  - Nsight Systems (权限要求较低)")
    print()
    
    print("注意：在生产环境中，建议只在需要时临时启用 profiling 权限。")

def run_pytorch_profiler_analysis(matrix_sizes: List[int], output_dir: Path):
    """使用 PyTorch Profiler 进行分析 (无权限要求)"""
    
    print("\n" + "="*50)
    print("使用 PyTorch Profiler 进行性能分析")
    print("="*50)
    
    # 加载扩展
    try:
        import matmul_cuda_ext
        print("[PASS] CUDA扩展加载成功")
    except Exception as e:
        print(f"[FAIL] CUDA扩展加载失败: {e}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # 定义实现
    implementations = {
        'pytorch_builtin': lambda a, b: torch.mm(a, b),
        'cuda_basic': lambda a, b: matmul_cuda_ext.matmul_basic(a, b),
        'cuda_shared': lambda a, b: matmul_cuda_ext.matmul_shared(a, b),
        'cuda_static_shared': lambda a, b: matmul_cuda_ext.matmul_static_shared(a, b),
        'cuda_template_32': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 32),
    }
    
    results = {}
    
    for size in matrix_sizes:
        print(f"\n分析矩阵大小: {size}x{size}")
        
        # 创建测试数据
        torch.manual_seed(42)
        A = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
        B = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
        
        size_results = {}
        
        for name, func in implementations.items():
            print(f"  分析 {name}...")
            
            # 预热
            for _ in range(3):
                _ = func(A, B)
                torch.cuda.synchronize()
            
            # 使用 PyTorch Profiler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for _ in range(10):
                    result = func(A, B)
                    torch.cuda.synchronize()
            
            # 保存 profile 结果
            profile_file = output_dir / f"{name}_{size}x{size}_pytorch_profile.json"
            prof.export_chrome_trace(str(profile_file))
            
            # 计算性能指标
            events = prof.events()
            cuda_events = [e for e in events if e.device_type == torch.profiler.DeviceType.CUDA]
            if cuda_events:
                avg_time = sum(e.cuda_time_total for e in cuda_events) / len(cuda_events) / 1000  # 转换为毫秒
                gflops = (2 * size ** 3) / (avg_time * 1e-3) / 1e9  # GFLOPS
                
                size_results[name] = {
                    'avg_time_ms': avg_time,
                    'gflops': gflops,
                    'profile_file': str(profile_file)
                }
                
                print(f"    平均时间: {avg_time:.3f} ms")
                print(f"    性能: {gflops:.1f} GFLOPS")
        
        results[size] = size_results
    
    # 生成报告
    report_file = output_dir / "pytorch_profiler_report.md"
    generate_pytorch_profiler_report(results, report_file)
    
    print(f"\n[PASS] PyTorch Profiler 分析完成")
    print(f"结果保存在: {output_dir}/")
    print(f"查看报告: cat {report_file}")
    print("使用 Chrome 浏览器打开 .json 文件查看详细的 profiling 时间线")

def generate_pytorch_profiler_report(results: Dict, report_file: Path):
    """生成 PyTorch Profiler 报告"""
    
    content = f'''# CUDA MatMul PyTorch Profiler 分析报告

## 环境信息
- 设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch 版本: {torch.__version__}
- 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 性能分析结果

'''
    
    for size, size_results in results.items():
        content += f"### 矩阵大小: {size}x{size}\n\n"
        content += "| 实现 | 平均时间 (ms) | 性能 (GFLOPS) | Profile 文件 |\n"
        content += "|------|---------------|---------------|-------------|\n"
        
        for impl_name, metrics in size_results.items():
            content += f"| {impl_name} | {metrics['avg_time_ms']:.3f} | {metrics['gflops']:.1f} | {Path(metrics['profile_file']).name} |\n"
        
        content += "\n"
    
    content += '''## 使用 PyTorch Profiler 的优势

1. **无权限要求**: 不需要特殊的系统权限
2. **详细的时间线**: 可以看到 CPU 和 GPU 的执行时间线
3. **内存分析**: 包含内存使用情况分析
4. **易于集成**: 直接集成在 PyTorch 代码中

## 查看详细结果

### Chrome Tracing 可视化
```bash
# 在 Chrome 浏览器中打开 chrome://tracing/
# 然后加载 .json 文件查看详细的执行时间线
```

### 命令行分析
```python
import torch
prof = torch.profiler.profile(...)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 与 Nsight Compute 的对比

| 特性 | PyTorch Profiler | Nsight Compute |
|------|------------------|----------------|
| 权限要求 | 无 | 需要管理员权限 |
| 分析粒度 | 函数级 | 指令级 |
| 内存分析 | 基本 | 详细 |
| 可视化 | Chrome Tracing | 专业 GUI |
| 使用难度 | 简单 | 复杂 |

## 优化建议

基于 PyTorch Profiler 的分析结果：
1. 关注 CUDA kernel 的执行时间
2. 查看内存分配和释放模式
3. 分析 CPU-GPU 同步开销
4. 对比不同实现的时间线差异
'''
    
    with open(report_file, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='CUDA MatMul 性能分析 - 权限友好版本')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024],
                       help='矩阵大小列表 (默认: [1024])')
    parser.add_argument('--output', '-o', default='profiling_analysis',
                       help='输出目录 (默认: profiling_analysis)')
    parser.add_argument('--force-pytorch-profiler', action='store_true',
                       help='强制使用 PyTorch Profiler')
    
    args = parser.parse_args()
    
    print("CUDA MatMul 性能分析工具")
    print("=" * 50)
    print(f"矩阵大小: {args.sizes}")
    print(f"输出目录: {args.output}")
    print()
    
    output_dir = Path(args.output)
    
    # 检查 NCU 权限
    if not args.force_pytorch_profiler:
        print("检查 Nsight Compute 权限...")
        if check_ncu_permissions():
            print("[PASS] Nsight Compute 权限正常，可以使用 ncu_profiler.py")
            print("运行: python ncu_profiler.py --sizes", ' '.join(map(str, args.sizes)))
            return 0
        else:
            print("[FAIL] Nsight Compute 权限不足")
            provide_permission_guidance()
            
            print("\n使用 PyTorch Profiler 作为替代方案...")
    
    # 使用 PyTorch Profiler
    run_pytorch_profiler_analysis(args.sizes, output_dir)
    
    return 0

if __name__ == "__main__":
    exit(main())
