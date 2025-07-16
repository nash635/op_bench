#!/usr/bin/env python3
"""
使用 Nsight Systems 对 CUDA MatMul 算子进行性能 profiling
"""

import torch
import time
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# 设置PyTorch库路径以解决CUDA扩展导入问题
def setup_library_path():
    """设置正确的库路径以确保CUDA扩展能够正常加载"""
    torch_lib_path = os.path.join(torch.__path__[0], 'lib')
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    if torch_lib_path not in current_ld_path:
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = torch_lib_path
        print(f"[INFO] 设置库路径: {torch_lib_path}")

# 在导入其他模块前设置库路径
setup_library_path()

class NsightProfiler:
    def __init__(self, output_dir: str = "nsight_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matmul_cuda = None
        
        # 延迟加载CUDA扩展
        self._load_cuda_extension()
    
    def _load_cuda_extension(self):
        """加载CUDA扩展"""
        try:
            import matmul_cuda_ext
            self.matmul_cuda = matmul_cuda_ext
            print("[PASS] CUDA扩展加载成功")
        except Exception as e:
            print(f"[FAIL] CUDA扩展加载失败: {e}")
            self.matmul_cuda = None
    
    def check_nsight_availability(self) -> bool:
        """检查 Nsight Systems 是否可用"""
        try:
            result = subprocess.run(['nsys', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[PASS] Nsight Systems 可用: {result.stdout.strip()}")
                return True
            else:
                print("[FAIL] Nsight Systems 不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[FAIL] 未找到 nsys 命令，请确保 Nsight Systems 已安装并在 PATH 中")
            return False
    
    def check_nsight_compute_availability(self) -> bool:
        """检查 Nsight Compute 是否可用"""
        try:
            result = subprocess.run(['ncu', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[PASS] Nsight Compute 可用: {result.stdout.strip().split()[0:3]}")
                return True
            else:
                print("[FAIL] Nsight Compute 不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[FAIL] 未找到 ncu 命令，请确保 Nsight Compute 已安装并在 PATH 中")
            return False
    
    def create_test_matrices(self, m: int, n: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建测试矩阵"""
        torch.manual_seed(42)
        A = torch.randn(m, k, device=self.device, dtype=torch.float32).contiguous()
        B = torch.randn(k, n, device=self.device, dtype=torch.float32).contiguous()
        return A, B
    
    def warmup_kernels(self, A: torch.Tensor, B: torch.Tensor, implementations: Dict):
        """预热所有 kernel"""
        print("预热 kernels...")
        for name, func in implementations.items():
            try:
                for _ in range(3):
                    _ = func(A, B)
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"预热 {name} 失败: {e}")
    
    def profile_single_kernel(self, kernel_name: str, kernel_func, A: torch.Tensor, B: torch.Tensor, 
                             matrix_size: str, iterations: int = 100) -> str:
        """对单个 kernel 进行 profiling"""
        
        # 生成临时 Python 脚本用于 profiling
        temp_script = self.output_dir / f"temp_profile_{kernel_name}.py"
        output_file = self.output_dir / f"{kernel_name}_{matrix_size}_profile.nsys-rep"
        
        # 写入临时脚本
        script_content = f'''
import sys
import os
# 添加当前目录到 Python 路径以找到编译的扩展
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "{os.getcwd()}")

import torch
import matmul_cuda_ext
import time

# 重新创建相同的矩阵
torch.manual_seed(42)
A = torch.randn({A.shape[0]}, {A.shape[1]}, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn({B.shape[0]}, {B.shape[1]}, device='cuda', dtype=torch.float32).contiguous()

# 选择对应的函数
implementations = {{
    'pytorch_builtin': lambda a, b: torch.mm(a, b),
    'cuda_basic': lambda a, b: matmul_cuda_ext.matmul_basic(a, b),
    'cuda_shared': lambda a, b: matmul_cuda_ext.matmul_shared(a, b),
    'cuda_static_shared': lambda a, b: matmul_cuda_ext.matmul_static_shared(a, b),
    'cuda_template_8': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 8),
    'cuda_template_16': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 16),
    'cuda_template_32': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 32),
}}

func = implementations['{kernel_name}']

# 预热
for _ in range(5):
    _ = func(A, B)
    torch.cuda.synchronize()

# 开始 profiling 区域
torch.cuda.nvtx.range_push("MatMul_Profiling_{kernel_name}")

# 运行多次迭代
for i in range({iterations}):
    torch.cuda.nvtx.range_push(f"Iteration_{{i}}")
    result = func(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_pop()
'''
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # 构建 nsys 命令 (适配较老的GPU，移除不兼容的选项)
        nsys_cmd = [
            'nsys', 'profile',
            '--output', str(output_file),
            '--force-overwrite', 'true',
            '--trace', 'cuda,nvtx',
            '--cuda-memory-usage', 'true',
            '--duration', '30',  # 最多 profiling 30 秒
            'python', str(temp_script)
        ]
        
        print(f"  Profiling {kernel_name}...")
        try:
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"    [PASS] Profile 保存至: {output_file}")
                # 清理临时文件
                temp_script.unlink()
                return str(output_file)
            else:
                print(f"    [FAIL] Profiling 失败: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] Profiling 超时")
            return ""
        except Exception as e:
            print(f"    [FAIL] Profiling 异常: {e}")
            return ""
    
    def profile_kernel_with_ncu(self, kernel_name: str, A: torch.Tensor, B: torch.Tensor,
                               matrix_size: str, metrics: List[str] = None) -> str:
        """使用 Nsight Compute 对单个 kernel 进行详细 profiling"""
        
        if metrics is None:
            # 默认的重要性能指标
            metrics = [
                'sm__cycles_elapsed.avg',
                'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                'gpu__time_duration.sum',
                'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
                'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
                'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
                'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'
            ]
        
        # 生成临时 Python 脚本用于 NCU profiling
        temp_script = self.output_dir / f"temp_ncu_{kernel_name}.py"
        output_file = self.output_dir / f"{kernel_name}_{matrix_size}_ncu_profile.ncu-rep"
        
        # 写入临时脚本
        script_content = f'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "{os.getcwd()}")

import torch
import matmul_cuda_ext

# 重新创建相同的矩阵
torch.manual_seed(42)
A = torch.randn({A.shape[0]}, {A.shape[1]}, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn({B.shape[0]}, {B.shape[1]}, device='cuda', dtype=torch.float32).contiguous()

# 选择对应的函数
implementations = {{
    'pytorch_builtin': lambda a, b: torch.mm(a, b),
    'cuda_basic': lambda a, b: matmul_cuda_ext.matmul_basic(a, b),
    'cuda_shared': lambda a, b: matmul_cuda_ext.matmul_shared(a, b),
    'cuda_static_shared': lambda a, b: matmul_cuda_ext.matmul_static_shared(a, b),
    'cuda_template_8': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 8),
    'cuda_template_16': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 16),
    'cuda_template_32': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 32),
}}

func = implementations['{kernel_name}']

# 预热
for _ in range(3):
    _ = func(A, B)
    torch.cuda.synchronize()

# 执行用于 profiling 的 kernel
result = func(A, B)
torch.cuda.synchronize()
'''
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # 构建 ncu 命令
        metrics_args = ['--metrics', ','.join(metrics)]  # 将所有指标用逗号连接
        
        ncu_cmd = [
            'ncu',
            '--target-processes', 'all',
            '--launch-count', '1',
            '--launch-skip', '0',
            '--export', str(output_file),
            '--force-overwrite'
        ] + metrics_args + [
            'python', str(temp_script)
        ]
        
        print(f"  NCU Profiling {kernel_name}...")
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"    [PASS] NCU Profile 保存至: {output_file}")
                # 清理临时文件
                temp_script.unlink()
                return str(output_file)
            else:
                print(f"    [FAIL] NCU Profiling 失败: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] NCU Profiling 超时")
            return ""
        except Exception as e:
            print(f"    [FAIL] NCU Profiling 异常: {e}")
            return ""

    def profile_all_kernels(self, matrix_sizes: List[Tuple[int, int, int]], 
                           iterations: int = 100) -> Dict[str, List[str]]:
        """对所有 kernel 进行 profiling"""
        
        if not self.check_nsight_availability():
            return {}
        
        if not self.matmul_cuda:
            print("CUDA 扩展未加载，无法进行 profiling")
            return {}
        
        # 定义所有实现
        implementations = {
            'pytorch_builtin': lambda a, b: torch.mm(a, b),
            'cuda_basic': lambda a, b: self.matmul_cuda.matmul_basic(a, b),
            'cuda_shared': lambda a, b: self.matmul_cuda.matmul_shared(a, b),
            'cuda_static_shared': lambda a, b: self.matmul_cuda.matmul_static_shared(a, b),
            'cuda_template_8': lambda a, b: self.matmul_cuda.matmul_template(a, b, 8),
            'cuda_template_16': lambda a, b: self.matmul_cuda.matmul_template(a, b, 16),
            'cuda_template_32': lambda a, b: self.matmul_cuda.matmul_template(a, b, 32),
        }
        
        profile_files = {}
        
        for m, k, n in matrix_sizes:
            print(f"\n=== Profiling 矩阵大小: ({m}, {k}) × ({k}, {n}) ===")
            matrix_size_str = f"{m}x{k}x{n}"
            
            A, B = self.create_test_matrices(m, n, k)
            self.warmup_kernels(A, B, implementations)
            
            for kernel_name in implementations.keys():
                if kernel_name not in profile_files:
                    profile_files[kernel_name] = []
                
                profile_file = self.profile_single_kernel(
                    kernel_name, implementations[kernel_name], 
                    A, B, matrix_size_str, iterations
                )
                
                if profile_file:
                    profile_files[kernel_name].append(profile_file)
                
                # 短暂休息避免 GPU 过热
                time.sleep(1)
        
        return profile_files
    
    def profile_all_kernels_ncu(self, matrix_sizes: List[Tuple[int, int, int]], 
                               kernel_names: List[str] = None) -> Dict[str, List[str]]:
        """使用 Nsight Compute 对指定 kernel 进行详细 profiling"""
        
        if not self.check_nsight_compute_availability():
            return {}
        
        if not self.matmul_cuda:
            print("CUDA 扩展未加载，无法进行 NCU profiling")
            return {}
        
        if kernel_names is None:
            # 默认 profile 几个主要的 kernel
            kernel_names = ['pytorch_builtin', 'cuda_basic', 'cuda_shared', 'cuda_template_32']
        
        profile_files = {}
        
        for m, k, n in matrix_sizes:
            print(f"\n=== NCU Profiling 矩阵大小: ({m}, {k}) × ({k}, {n}) ===")
            matrix_size_str = f"{m}x{k}x{n}"
            
            A, B = self.create_test_matrices(m, n, k)
            
            for kernel_name in kernel_names:
                if kernel_name not in profile_files:
                    profile_files[kernel_name] = []
                
                profile_file = self.profile_kernel_with_ncu(
                    kernel_name, A, B, matrix_size_str
                )
                
                if profile_file:
                    profile_files[kernel_name].append(profile_file)
                
                # 休息一下避免 GPU 过热
                time.sleep(2)
        
        return profile_files

    def generate_analysis_script(self, profile_files: Dict[str, List[str]]):
        """生成分析 profile 文件的脚本"""
        analysis_script = self.output_dir / "analyze_profiles.sh"
        
        script_content = '''#!/bin/bash
# Nsight Systems Profile 分析脚本
# 使用方法: ./analyze_profiles.sh

echo "=== Nsight Systems Profile 分析 ==="
echo

# 检查 nsys 是否可用
if ! command -v nsys &> /dev/null; then
    echo "错误: nsys 命令未找到"
    exit 1
fi

'''
        
        # 为每个 profile 文件添加分析命令
        for kernel_name, files in profile_files.items():
            script_content += f'\necho "=== 分析 {kernel_name} ==="\\n'
            for profile_file in files:
                if os.path.exists(profile_file):
                    profile_basename = os.path.basename(profile_file)
                    script_content += f'''
echo "Profile 文件: {profile_basename}"
echo "生成统计报告..."
nsys stats --report gputrace,gpukernsum,gpumemtimesum {profile_basename}
echo
'''
        
        script_content += r'''
echo "=== 生成可视化报告 ==="
echo "可以使用 Nsight Systems GUI 打开 .nsys-rep 文件进行详细分析"
echo "或使用以下命令生成 SQLite 数据库:"
echo

for file in *.nsys-rep; do
    if [ -f "$file" ]; then
        echo "nsys export --type sqlite --output ${file%.nsys-rep}.sqlite $file"
    fi
done
'''
        
        with open(analysis_script, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(analysis_script, 0o755)
        print(f"\n[PASS] 分析脚本已生成: {analysis_script}")
    
    def generate_ncu_analysis_script(self, ncu_profile_files: Dict[str, List[str]]):
        """生成 Nsight Compute 分析脚本"""
        analysis_script = self.output_dir / "analyze_ncu_profiles.sh"
        
        script_content = '''#!/bin/bash
# Nsight Compute Profile 分析脚本
# 使用方法: ./analyze_ncu_profiles.sh

echo "=== Nsight Compute Profile 分析 ==="
echo

# 检查 ncu 是否可用
if ! command -v ncu &> /dev/null; then
    echo "错误: ncu 命令未找到"
    exit 1
fi

'''
        
        # 为每个 NCU profile 文件添加分析命令
        for kernel_name, files in ncu_profile_files.items():
            script_content += f'\necho "=== NCU 分析 {kernel_name} ==="\\n'
            for profile_file in files:
                if os.path.exists(profile_file):
                    profile_basename = os.path.basename(profile_file)
                    script_content += f'''
echo "NCU Profile 文件: {profile_basename}"
echo "生成详细报告..."
ncu --import {profile_basename} --print-summary
echo
echo "生成 CSV 报告..."
ncu --import {profile_basename} --csv > "${{profile_basename%.ncu-rep}}_summary.csv"
echo
'''
        
        script_content += r'''
echo "=== 生成对比报告 ==="
echo "可以使用 Nsight Compute GUI 打开 .ncu-rep 文件进行详细分析"
echo "或使用以下命令导出详细数据:"
echo

for file in *.ncu-rep; do
    if [ -f "$file" ]; then
        echo "ncu --import $file --csv > ${file%.ncu-rep}_detailed.csv"
    fi
done
'''
        
        with open(analysis_script, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(analysis_script, 0o755)
        print(f"\n[PASS] NCU 分析脚本已生成: {analysis_script}")
    
    def generate_summary_report(self, profile_files: Dict[str, List[str]]):
        """生成 profiling 总结报告"""
        report_file = self.output_dir / "profiling_summary.md"
        
        content = f'''# CUDA MatMul Nsight Systems Profiling 报告

## 环境信息
- 设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch 版本: {torch.__version__}
- 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Profile 文件列表

'''
        
        for kernel_name, files in profile_files.items():
            content += f"### {kernel_name}\n\n"
            for profile_file in files:
                if os.path.exists(profile_file):
                    file_size = os.path.getsize(profile_file) / 1024 / 1024  # MB
                    content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"
            content += "\n"
        
        content += '''## 使用方法

### 1. 命令行分析
```bash
# 运行分析脚本
./analyze_profiles.sh

# 或手动分析单个文件
nsys stats --report gputrace,gpukernsum,gpumemtimesum <profile_file.nsys-rep>
```

### 2. GUI 分析
```bash
# 启动 Nsight Systems GUI
nsight-sys

# 然后打开 .nsys-rep 文件进行可视化分析
```

### 3. 导出数据
```bash
# 导出为 SQLite 数据库
nsys export --type sqlite --output profile_data.sqlite <profile_file.nsys-rep>
```

## 关键分析指标

1. **Kernel 执行时间**: GPU kernel 的实际执行时间
2. **内存传输**: Host-Device 数据传输时间
3. **GPU 利用率**: GPU 计算单元的利用率
4. **内存带宽**: 内存访问效率
5. **线程占用率**: SM 的线程占用情况

## 优化建议

基于 profiling 结果，可以关注以下优化方向:
- Kernel 启动开销
- 内存访问模式
- 线程块大小
- 共享内存使用效率
- 寄存器使用情况
'''
        
        with open(report_file, 'w') as f:
            f.write(content)
        
        print(f"[PASS] 总结报告已生成: {report_file}")
    
    def generate_comprehensive_report(self, nsys_files: Dict[str, List[str]], 
                                     ncu_files: Dict[str, List[str]], mode: str):
        """生成综合性能分析报告"""
        report_file = self.output_dir / "comprehensive_profiling_report.md"
        
        content = f'''# CUDA MatMul 综合性能分析报告

## 环境信息
- 设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch 版本: {torch.__version__}
- 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
- Profiling 模式: {mode}

## 分析工具对比

### Nsight Systems vs Nsight Compute

| 工具 | 用途 | 分析粒度 | 主要指标 |
|------|------|----------|----------|
| Nsight Systems | 系统级性能分析 | 整个应用程序 | 时间线、GPU利用率、内存传输 |
| Nsight Compute | Kernel级详细分析 | 单个CUDA kernel | 指令吞吐量、内存效率、占用率 |

'''
        
        if nsys_files:
            content += '''## Nsight Systems 分析结果

### 时间线分析
Nsight Systems 提供了完整的执行时间线，可以分析：
- Kernel 启动开销
- 内存传输时间
- CPU-GPU 同步点
- 整体程序流程

### Profile 文件列表
'''
            for kernel_name, files in nsys_files.items():
                content += f"#### {kernel_name}\n"
                for profile_file in files:
                    if os.path.exists(profile_file):
                        file_size = os.path.getsize(profile_file) / 1024 / 1024
                        content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"
                content += "\n"
        
        if ncu_files:
            content += '''## Nsight Compute 分析结果

### 详细 Kernel 分析
Nsight Compute 提供了每个 kernel 的详细性能指标：
- 指令级吞吐量分析
- 内存子系统效率
- 寄存器和共享内存使用
- 线程占用率和延迟分析

### Profile 文件列表
'''
            for kernel_name, files in ncu_files.items():
                content += f"#### {kernel_name}\n"
                for profile_file in files:
                    if os.path.exists(profile_file):
                        file_size = os.path.getsize(profile_file) / 1024 / 1024
                        content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"
                content += "\n"
        
        content += '''## 分析工作流程

### 1. 系统级分析 (Nsight Systems)
```bash
# 运行 Nsight Systems 分析
./analyze_profiles.sh

# 或手动分析
nsys stats --report gputrace,gpukernsum profile.nsys-rep
```

### 2. Kernel级分析 (Nsight Compute)
```bash
# 运行 Nsight Compute 分析
./analyze_ncu_profiles.sh

# 或手动分析
ncu --import profile.ncu-rep --print-summary
```

### 3. GUI 可视化分析
```bash
# Nsight Systems GUI
nsight-sys *.nsys-rep

# Nsight Compute GUI  
ncu-ui *.ncu-rep
```

## 关键性能指标

### Nsight Systems 指标
1. **GPU 利用率**: 总体 GPU 使用率
2. **内存带宽**: DRAM 吞吐量
3. **Kernel 执行时间**: 各 kernel 的执行时间对比
4. **启动开销**: Kernel 启动和调度时间

### Nsight Compute 指标
1. **计算吞吐量**: 
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
   - 各类浮点运算的执行效率

2. **内存效率**:
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`

3. **占用率**:
   - `sm__warps_active.avg.pct_of_peak_sustained_active`
   - 寄存器和共享内存使用情况

## 优化建议

### 基于 Nsight Systems 的优化
1. **减少 Kernel 启动开销**: 合并小的 kernel 调用
2. **优化内存传输**: 减少 Host-Device 数据传输
3. **改善 GPU 利用率**: 确保 GPU 持续有工作负载

### 基于 Nsight Compute 的优化
1. **提高计算吞吐量**: 
   - 优化指令混合 (指令级并行)
   - 减少分支分歧

2. **优化内存访问**:
   - 改善 memory coalescing
   - 增加 cache 命中率
   - 合理使用共享内存

3. **提高占用率**:
   - 调整线程块大小
   - 优化寄存器使用
   - 平衡共享内存使用

## 工具使用技巧

### Nsight Systems
- 使用 NVTX 标记关键代码段
- 分析不同矩阵大小的性能缩放
- 关注 GPU 空闲时间和瓶颈

### Nsight Compute  
- 重点关注性能限制因子 (compute vs memory bound)
- 使用 roofline 模型分析理论性能上限
- 对比不同优化版本的详细指标

## 下一步分析

1. **运行分析脚本**: 获取量化的性能数据
2. **GUI 深入分析**: 使用可视化工具探索性能瓶颈  
3. **对比分析**: 比较不同实现的关键指标
4. **迭代优化**: 基于分析结果调整 kernel 实现
'''
        
        with open(report_file, 'w') as f:
            f.write(content)
        
        print(f"[PASS] 综合分析报告已生成: {report_file}")
def main():
    parser = argparse.ArgumentParser(description='CUDA MatMul Nsight Systems/Compute Profiling')
    parser.add_argument('--output', '-o', default='nsight_profiles', 
                       help='输出目录 (默认: nsight_profiles)')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='每个 kernel 的迭代次数 (默认: 100)')
    parser.add_argument('--sizes', nargs='+', default=['1024'],
                       help='矩阵大小列表 (默认: [1024])')
    parser.add_argument('--kernels', nargs='+', 
                       default=['pytorch_builtin', 'cuda_template_32', 'cuda_static_shared'],
                       help='要 profile 的 kernel 列表')
    parser.add_argument('--mode', choices=['nsys', 'ncu', 'both'], default='nsys',
                       help='Profiling 模式: nsys (Nsight Systems), ncu (Nsight Compute), both (默认: nsys)')
    
    # 添加与 README 文档一致的参数
    parser.add_argument('--operator', default='matmul',
                       help='要分析的算子类型 (默认: matmul)')
    parser.add_argument('--test-case', default='large_square',
                       help='测试用例名称 (默认: large_square)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Profiling 持续时间（秒） (默认: 10)')
    
    args = parser.parse_args()
    
    # 解析矩阵大小
    matrix_sizes = []
    for size_str in args.sizes:
        try:
            size = int(size_str)
            matrix_sizes.append((size, size, size))  # 方阵
        except ValueError:
            print(f"无效的矩阵大小: {size_str}")
            continue
    
    if not matrix_sizes:
        matrix_sizes = [(1024, 1024, 1024)]  # 默认
    
    print("CUDA MatMul Nsight Profiler")
    print("=" * 50)
    print(f"算子类型: {args.operator}")
    print(f"测试用例: {args.test_case}")
    print(f"输出目录: {args.output}")
    print(f"迭代次数: {args.iterations}")
    print(f"持续时间: {args.duration}s")
    print(f"矩阵大小: {matrix_sizes}")
    print(f"Profile kernels: {args.kernels}")
    print(f"Profiling 模式: {args.mode}")
    print()
    
    profiler = NsightProfiler(args.output)
    
    nsys_profile_files = {}
    ncu_profile_files = {}
    
    # 运行 Nsight Systems profiling
    if args.mode in ['nsys', 'both']:
        print("=== 运行 Nsight Systems Profiling ===")
        nsys_profile_files = profiler.profile_all_kernels(matrix_sizes, args.iterations)
        
        if nsys_profile_files:
            profiler.generate_analysis_script(nsys_profile_files)
    
    # 运行 Nsight Compute profiling
    if args.mode in ['ncu', 'both']:
        print("\n=== 运行 Nsight Compute Profiling ===")
        ncu_profile_files = profiler.profile_all_kernels_ncu(matrix_sizes, args.kernels)
        
        if ncu_profile_files:
            profiler.generate_ncu_analysis_script(ncu_profile_files)
    
    # 生成综合报告
    if nsys_profile_files or ncu_profile_files:
        profiler.generate_comprehensive_report(nsys_profile_files, ncu_profile_files, args.mode)
        
        print("\n" + "=" * 50)
        print("Profiling 完成!")
        print(f"结果保存在: {args.output}/")
        print("下一步:")
        
        if nsys_profile_files:
            print(f"  1. Nsight Systems: cd {args.output} && ./analyze_profiles.sh")
        if ncu_profile_files:
            print(f"  2. Nsight Compute: cd {args.output} && ./analyze_ncu_profiles.sh")
        
        print(f"  3. 或使用 GUI 打开 profile 文件进行详细分析")
    else:
        print("Profiling 失败，请检查环境配置")

if __name__ == "__main__":
    main()
