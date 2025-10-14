
#!/usr/bin/env python3
"""
Validate correctness and basic functionality of the optimal_matmul_cuda extension.
"""

import torch
import time
import sys
import os

def test_correctness():
    """Validate algorithm correctness."""
    print("=" * 60)
    print("Correctness test")
    print("=" * 60)
    
    # Import the extension
    try:
        from torch.utils.cpp_extension import load
        matmul_cuda = load(
            name='optimal_matmul_cuda',
            sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
            verbose=False
        )
        print("[PASS] Extension loaded successfully")
    except Exception as e:
        print(f"[FAIL] Extension load failed: {e}")
        return False
    
    # Create test matrices
    torch.manual_seed(42)
    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    for m, k, n in sizes:
        print(f"\nTesting matrix dims: ({m}, {k}) × ({k}, {n})")
        
        A = torch.randn(m, k, device='cuda', dtype=torch.float32).contiguous()
        B = torch.randn(k, n, device='cuda', dtype=torch.float32).contiguous()
        reference = torch.mm(A, B)
        
        # Evaluate all implementations
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
                status = "[PASS]" if is_correct else "[FAIL]"
                print(f"  {status} {name}: {'correct' if is_correct else 'incorrect'}")
                if not is_correct:
                    max_diff = torch.max(torch.abs(result - reference)).item()
                    print(f"    Max difference: {max_diff}")
                    all_correct = False
            except Exception as e:
                print(f"  [FAIL] {name}: exception - {e}")
                all_correct = False
        
        if not all_correct:
            return False
    
    print("\n[PASS] All correctness tests passed!")
    return True

def test_performance():
    """Basic performance comparison."""
    print("\n" + "=" * 60)
    print("Performance smoke test")
    print("=" * 60)
    
    # Import the extension
    from torch.utils.cpp_extension import load
    matmul_cuda = load(
        name='optimal_matmul_cuda',
        sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        verbose=False
    )
    
    # Benchmark matrix
    size = 1024
    A = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
    
    # Warmup
    for _ in range(3):
        _ = torch.mm(A, B)
        _ = matmul_cuda.matmul_static_shared(A, B)
        torch.cuda.synchronize()
    
    # Measure PyTorch baseline
    start = time.time()
    for _ in range(10):
        result_pytorch = torch.mm(A, B)
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10
    
    # Measure our implementation
    start = time.time()
    for _ in range(10):
        result_ours = matmul_cuda.matmul_static_shared(A, B)
        torch.cuda.synchronize()
    our_time = (time.time() - start) / 10
    
    # Compute GFLOPS
    flops = 2 * size**3
    pytorch_gflops = flops / (pytorch_time * 1e9)
    our_gflops = flops / (our_time * 1e9)
    speedup = pytorch_time / our_time
    
    print(f"Matrix size: {size} × {size}")
    print(f"PyTorch:     {pytorch_time*1000:.2f} ms ({pytorch_gflops:.1f} GFLOPS)")
    print(f"Custom impl: {our_time*1000:.2f} ms ({our_gflops:.1f} GFLOPS)")
    print(f"Speedup:     {speedup:.2f}x")

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print("PyTorch CUDA MatMul extension test suite")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Run correctness test
    if test_correctness():
        # Followed by performance test
        test_performance()
    else:
        print("Correctness test failed; skipping performance run")

if __name__ == "__main__":
    main()
