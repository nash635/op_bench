#!/usr/bin/env python3
"""
CUDA MatMul profiling helper for permission-restricted environments.
Provides alternative analysis paths when Nsight Compute access is limited.
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
    """Check whether Nsight Compute can access performance counters."""
    try:
        # Create a lightweight test script
        test_script = Path("temp_permission_test.py")
        test_script.write_text('''
import torch
if torch.cuda.is_available():
    a = torch.randn(2, 2, device='cuda')
    print("Simple CUDA test completed")
''')
        
        # Attempt to run NCU
        result = subprocess.run([
            'ncu', '--target-processes', 'all', '--launch-count', '1',
            '--export', 'temp_test.ncu-rep', '--force-overwrite',
            'python', str(test_script)
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up temporary files
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
    """Offer guidance for configuring profiling permissions."""
    print("\n" + "="*60)
    print("Nsight Compute permission guidance")
    print("="*60)
    print("\nNsight Compute needs elevated permissions to access GPU performance counters.")
    print("Try the following options:\n")
    
    print("Option 1: Temporarily run with sudo")
    print("  sudo python ncu_profiler.py --sizes 512")
    print()
    
    print("Option 2: Adjust driver parameters (requires reboot)")
    print("  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf")
    print("  sudo update-initramfs -u")
    print("  sudo reboot")
    print()
    
    print("Option 3: Use a Docker container")
    print("  nvidia-docker run --privileged -it nvidia/cuda:12.0-devel")
    print()
    
    print("Option 4: Switch to alternative profiling approaches")
    print("  - PyTorch Profiler (no extra permissions)")
    print("  - Manual timing with GFLOPS calculation")
    print("  - Nsight Systems (lower permission requirements)")
    print()
    
    print("Note: In production, enable profiling permissions only when necessary.")

def run_pytorch_profiler_analysis(matrix_sizes: List[int], output_dir: Path):
    """Profile CUDA kernels using PyTorch Profiler without special permissions."""
    
    print("\n" + "="*50)
    print("Profiling performance with PyTorch Profiler")
    print("="*50)
    
    # Load the CUDA extension if possible
    matmul_cuda_ext = None
    try:
        # First try direct import
        import matmul_cuda_ext
        print("[PASS] CUDA extension loaded via direct import")
    except ImportError:
        try:
            # Try JIT compilation with correct paths
            import os
            import sys
            from torch.utils.cpp_extension import load
            
            # Get the correct source file paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cuda_dir = os.path.join(project_root, 'src', 'cuda')
            
            sources = [
                os.path.join(cuda_dir, 'matmul_cuda_ext.cpp'),
                os.path.join(cuda_dir, 'matmul_kernels.cu')
            ]
            
            # Check if source files exist
            if not all(os.path.exists(src) for src in sources):
                print(f"[WARN] CUDA source files not found: {sources}")
                print("[INFO] Falling back to the PyTorch built-in implementation only")
            else:
                matmul_cuda_ext = load(
                    name='matmul_cuda_ext',
                    sources=sources,
                    extra_cflags=['-O3', '-std=c++17'],
                    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
                    verbose=False
                )
                print("[PASS] CUDA extension loaded via JIT compilation")
        except Exception as e:
            print(f"[WARN] Failed to load CUDA extension: {e}")
            print("[INFO] Only the PyTorch built-in implementation will be profiled")
            matmul_cuda_ext = None
    
    output_dir.mkdir(exist_ok=True)
    
    # Register implementations to evaluate
    implementations = {
        'pytorch_builtin': lambda a, b: torch.mm(a, b),
    }
    
    # Add CUDA implementations if extension is available
    if matmul_cuda_ext is not None:
        implementations.update({
            'cuda_basic': lambda a, b: matmul_cuda_ext.matmul_basic(a, b),
            'cuda_shared': lambda a, b: matmul_cuda_ext.matmul_shared(a, b),
            'cuda_static_shared': lambda a, b: matmul_cuda_ext.matmul_static_shared(a, b),
            'cuda_template_16': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 16),
            'cuda_template_32': lambda a, b: matmul_cuda_ext.matmul_template(a, b, 32),
        })
    
    results = {}
    
    for size in matrix_sizes:
        print(f"\nProfiling matrix size: {size}x{size}")
        
        # Create test data
        torch.manual_seed(42)
        A = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
        B = torch.randn(size, size, device='cuda', dtype=torch.float32).contiguous()
        
        size_results = {}
        
        for name, func in implementations.items():
            print(f"  Profiling {name}...")
            
            # Warmup iterations
            for _ in range(3):
                _ = func(A, B)
                torch.cuda.synchronize()
            
            # Measure with PyTorch Profiler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for _ in range(10):
                    result = func(A, B)
                    torch.cuda.synchronize()
            
            # Save profiler trace
            profile_file = output_dir / f"{name}_{size}x{size}_pytorch_profile.json"
            prof.export_chrome_trace(str(profile_file))
            
            # Compute basic performance metrics
            events = prof.events()
            cuda_events = [e for e in events if e.device_type == torch.profiler.DeviceType.CUDA]
            if cuda_events:
                avg_time = sum(e.cuda_time_total for e in cuda_events) / len(cuda_events) / 1000  # Convert to ms
                gflops = (2 * size ** 3) / (avg_time * 1e-3) / 1e9  # GFLOPS
                
                size_results[name] = {
                    'avg_time_ms': avg_time,
                    'gflops': gflops,
                    'profile_file': str(profile_file)
                }
                
                print(f"    Avg time: {avg_time:.3f} ms")
                print(f"    Throughput: {gflops:.1f} GFLOPS")
        
        results[size] = size_results
    
    # Generate report
    report_file = output_dir / "pytorch_profiler_report.md"
    generate_pytorch_profiler_report(results, report_file)
    
    print(f"\n[PASS] PyTorch Profiler run complete")
    print(f"Results saved in: {output_dir}/")
    print(f"View summary: cat {report_file}")
    print("Open the .json traces in Chrome's tracing viewer for detailed timelines")

def generate_pytorch_profiler_report(results: Dict, report_file: Path):
    """Generate a Markdown report summarizing PyTorch Profiler runs."""
    
    content = f'''# CUDA MatMul PyTorch Profiler Report

## Environment
- Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch version: {torch.__version__}
- Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Performance Results

'''
    
    for size, size_results in results.items():
        content += f"### Matrix size: {size}x{size}\n\n"
        content += "| Implementation | Avg Time (ms) | Throughput (GFLOPS) | Profile File |\n"
        content += "|----------------|---------------|---------------------|--------------|\n"

        for impl_name, metrics in size_results.items():
            content += (
                f"| {impl_name} | {metrics['avg_time_ms']:.3f} | "
                f"{metrics['gflops']:.1f} | {Path(metrics['profile_file']).name} |\n"
            )

        content += "\n"
    
    content += '''## Benefits of PyTorch Profiler

1. **No special permissions**: Works without elevated system access
2. **Detailed timelines**: Capture both CPU and GPU execution
3. **Memory insights**: Includes memory usage analysis
4. **Easy integration**: Built directly into PyTorch

## Inspecting Results

### Chrome Tracing visualization
```bash
# Open chrome://tracing/ in Google Chrome
# Load the .json trace to inspect execution timelines
```

### Command-line analysis
```python
import torch
prof = torch.profiler.profile(...)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Comparison with Nsight Compute

| Attribute | PyTorch Profiler | Nsight Compute |
|-----------|------------------|----------------|
| Permission requirement | None | Administrator privileges |
| Analysis granularity | Function level | Instruction level |
| Memory analysis | Basic | Detailed |
| Visualization | Chrome tracing | Dedicated GUI |
| Ease of use | Simple | Complex |

## Optimization tips

Based on PyTorch Profiler insights:
1. Focus on CUDA kernel execution time
2. Inspect memory allocation and release patterns
3. Evaluate CPU-GPU synchronization overheads
4. Compare execution timelines across implementations
'''
    
    with open(report_file, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='CUDA MatMul profiling helper (permission friendly)')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024],
                       help='List of square matrix sizes (default: [1024])')
    parser.add_argument('--output', '-o', default='profiling_analysis',
                       help='Output directory (default: profiling_analysis)')
    parser.add_argument('--force-pytorch-profiler', action='store_true',
                       help='Force PyTorch Profiler even when Nsight Compute is available')
    
    args = parser.parse_args()
    
    print("CUDA MatMul profiling toolkit")
    print("=" * 50)
    print(f"Matrix sizes: {args.sizes}")
    print(f"Output directory: {args.output}")
    print()
    
    output_dir = Path(args.output)
    
    # Check Nsight Compute permissions if applicable
    if not args.force_pytorch_profiler:
        print("Checking Nsight Compute permissions...")
        if check_ncu_permissions():
            print("[PASS] Nsight Compute permissions look good; you can use ncu_profiler.py")
            print("Run: python ncu_profiler.py --sizes", ' '.join(map(str, args.sizes)))
            return 0
        else:
            print("[FAIL] Nsight Compute lacks the required permissions")
            provide_permission_guidance()
            
            print("\nFalling back to PyTorch Profiler...")
    
    # Use PyTorch Profiler
    run_pytorch_profiler_analysis(args.sizes, output_dir)
    
    return 0

if __name__ == "__main__":
    exit(main())
