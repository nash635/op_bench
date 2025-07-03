
#!/usr/bin/env python3
"""
测试optimal_matmul_cuda扩展的正确性和基本功能
"""

import torch
import time
import sys
import os

def test_correctness():
    """测试算法正确性"""
    print("=" * 60)
    print("正确性测试")
    print("=" * 60)
    
    # 导入扩展
    try:
        from torch.utils.cpp_extension import load
        matmul_cuda = load(
            name='optimal_matmul_cuda',
            sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
            verbose=False
        )
        print("✓ 扩展加载成功")
    except Exception as e:
        print(f"✗ 扩展加载失败: {e}")
        return False
    
    # 创建测试矩阵
    torch.manual_seed(42)
    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    for m, k, n in sizes:
        print(f"\n测试矩阵大小: ({m}, {k}) × ({k}, {n})")
        
        A = torch.randn(m, k, device='cuda', dtype=torch.float32).contiguous()
        B = torch.randn(k, n, device='cuda', dtype=torch.float32).contiguous()
        reference = torch.mm(A, B)
        
        # 测试所有实现
        implementations = [
            ('Basic', lambda: matmul_cuda.matmul_basic(A, B)),
            ('Shared', lambda: matmul_cuda.matmul_shared(A, B)),
            ('Static Shared', lambda: matmul_cuda.matmul_static_shared(A, B)),
            ('Template-16', lambda: matmul_cuda.matmul_template(A, B, 16)),
        ]
        
        all_correct = True
        for name, func in implementations:
            try:
                result = func()
                is_correct = torch.allclose(result, reference, rtol=1e-4, atol=1e-4)
                status = "✓" if is_correct else "✗"
                print(f"  {status} {name}: {'正确' if is_correct else '错误'}")
                if not is_correct:
                    max_diff = torch.max(torch.abs(result - reference)).item()
                    print(f"    最大差异: {max_diff}")
                    all_correct = False
            except Exception as e:
                print(f"  ✗ {name}: 异常 - {e}")
                all_correct = False
        
        if not all_correct:
            return False
    
    print("\n✓ 所有测试通过!")
    return True

def test_performance():
    """基本性能测试"""
    print("\n" + "=" * 60)
    print("基本性能测试")
    print("=" * 60)
    
    # 导入扩展
    from torch.utils.cpp_extension import load
    matmul_cuda = load(
        name='optimal_matmul_cuda',
        sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        verbose=False
    )
    
    # 测试矩阵
    size = 1024
    A = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
    
    # 预热
    for _ in range(3):
        _ = torch.mm(A, B)
        _ = matmul_cuda.matmul_static_shared(A, B)
        torch.cuda.synchronize()
    
    # 测试PyTorch
    start = time.time()
    for _ in range(10):
        result_pytorch = torch.mm(A, B)
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    # 测试我们的实现
    start = time.time()
    for _ in range(10):
        result_ours = matmul_cuda.matmul_static_shared(A, B)
        torch.cuda.synchronize()
    our_time = (time.time() - start) / 10
    
    # 计算GFLOPS
    flops = 2 * size**3
    pytorch_gflops = flops / (pytorch_time * 1e9)
    our_gflops = flops / (our_time * 1e9)
    speedup = pytorch_time / our_time
    
    print(f"矩阵大小: {size} × {size}")
    print(f"PyTorch:     {pytorch_time*1000:.2f} ms ({pytorch_gflops:.1f} GFLOPS)")
    print(f"我们的实现:   {our_time*1000:.2f} ms ({our_gflops:.1f} GFLOPS)")
    print(f"加速比:      {speedup:.2f}x")

def main():
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    print("PyTorch CUDA MatMul 扩展测试")
    print(f"设备: {torch.cuda.get_device_name()}")
    
    # 正确性测试
    if test_correctness():
        # 性能测试
        test_performance()
    else:
        print("正确性测试失败，跳过性能测试")

if __name__ == "__main__":
    main()
