#!/usr/bin/env python3
"""
Nsight Compute Profiler for Optimal MatMul CUDA Extension

This module provides detailed kernel-level performance analysis using NVIDIA Nsight Compute.
Supports multiple metrics sets, kernel comparisons, and automatic report generation.

Usage:
    python ncu_profiler.py --sizes 1024 --kernels cuda_template_16 cuda_template_32
    python ncu_profiler.py --sizes 512 1024 --metrics all
    ./run_ncu_profiling.sh 1024  # convenience script
"""

import argparse
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import torch
from torch.utils.cpp_extension import load


class NCUProfiler:
    """Nsight Compute profiler for CUDA kernels."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the profiler."""
        self.output_dir = Path(output_dir) if output_dir else Path(f"ncu_profiles_{time.strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Available metrics sets
        self.metrics_sets = {
            'basic': ['sm__cycles_elapsed.avg', 'sm__throughput.avg.pct_of_peak_sustained_elapsed'],
            'memory': [
                'dram__bytes_read.sum', 'dram__bytes_write.sum',
                'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
                'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
                'dram__throughput.avg.pct_of_peak_sustained_elapsed'
            ],
            'compute': [
                'sm__sass_thread_inst_executed_op_fadd_pred_on.sum',
                'sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 
                'sm__sass_thread_inst_executed_op_ffma_pred_on.sum',
                'sm__throughput.avg.pct_of_peak_sustained_elapsed'
            ],
            'occupancy': [
                'sm__warps_active.avg.pct_of_peak_sustained_active',
                'sm__maximum_warps_per_active_cycle_pct',
                'achieved_occupancy'
            ],
            'all': [
                'sm__cycles_elapsed.avg',
                'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                'dram__bytes_read.sum', 'dram__bytes_write.sum',
                'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
                'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
                'sm__sass_thread_inst_executed_op_fadd_pred_on.sum',
                'sm__sass_thread_inst_executed_op_fmul_pred_on.sum',
                'sm__sass_thread_inst_executed_op_ffma_pred_on.sum',
                'sm__warps_active.avg.pct_of_peak_sustained_active',
                'achieved_occupancy'
            ]
        }
        
        self.kernels_info = {
            'pytorch_builtin': 'PyTorch built-in torch.mm',
            'cuda_basic': 'Basic CUDA implementation',
            'cuda_shared': 'Dynamic shared memory optimization',
            'cuda_static_shared': 'Static shared memory optimization',
            'cuda_template_8': 'Template kernel with tile_width=8',
            'cuda_template_16': 'Template kernel with tile_width=16',
            'cuda_template_32': 'Template kernel with tile_width=32'
        }
        
    def check_gpu_compatibility(self) -> Tuple[bool, str]:
        """检查 GPU 是否支持 NCU"""
        try:
            # 获取 GPU 信息
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False, "无法获取 GPU 信息"
            
            gpu_info = result.stdout.strip().split('\n')[0]  # 取第一个 GPU
            gpu_name, compute_cap = [x.strip() for x in gpu_info.split(',')]
            compute_cap_major = int(float(compute_cap))
            
            # NCU 支持 compute capability 7.0 及以上 (Volta 架构及以上)
            if compute_cap_major < 7:
                return False, f"GPU {gpu_name} (计算能力 {compute_cap}) 不支持 NCU。NCU 需要计算能力 7.0+ (Volta 架构及以上)"
            
            return True, f"GPU {gpu_name} (计算能力 {compute_cap}) 支持 NCU"
            
        except Exception as e:
            return False, f"GPU 兼容性检查失败: {e}"
        
    def check_ncu_permissions(self) -> bool:
        """Check if NCU can run without sudo."""
        try:
            result = subprocess.run(['ncu', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False
                
            # Try a simple test
            test_cmd = ['ncu', '--metrics', 'sm__cycles_elapsed.avg', '--target-processes', 'all', 
                       'python', '-c', 'import torch; torch.cuda.synchronize()']
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
            if 'ERR_NVGPUCTRPERM' in test_result.stderr:
                return False
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def load_extension(self):
        """Load the CUDA extension."""
        try:
            self.matmul_cuda = load(
                name='optimal_matmul_cuda',
                sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
                verbose=False
            )
            return True
        except Exception as e:
            print(f"Error loading CUDA extension: {e}")
            return False
    
    def run_kernel(self, kernel_name: str, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Run a specific kernel."""
        if kernel_name == 'pytorch_builtin':
            return torch.mm(A, B)
        elif kernel_name == 'cuda_basic':
            return self.matmul_cuda.matmul_basic(A, B)
        elif kernel_name == 'cuda_shared':
            return self.matmul_cuda.matmul_shared(A, B)
        elif kernel_name == 'cuda_static_shared':
            return self.matmul_cuda.matmul_static_shared(A, B)
        elif kernel_name == 'cuda_template_8':
            return self.matmul_cuda.matmul_template(A, B, 8)
        elif kernel_name == 'cuda_template_16':
            return self.matmul_cuda.matmul_template(A, B, 16)
        elif kernel_name == 'cuda_template_32':
            return self.matmul_cuda.matmul_template(A, B, 32)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
    
    def create_profile_script(self, kernel_name: str, matrix_size: Tuple[int, int, int], 
                            metrics: List[str], iterations: int = 10) -> str:
        """Create a script to profile a specific kernel."""
        M, K, N = matrix_size
        script_content = f'''
import torch
from torch.utils.cpp_extension import load

# Load extension
matmul_cuda = load(
    name='optimal_matmul_cuda',
    sources=['matmul_cuda_ext.cpp', 'matmul_kernels.cu'],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False
)

# Setup
torch.cuda.empty_cache()
A = torch.randn({M}, {K}, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn({K}, {N}, device='cuda', dtype=torch.float32).contiguous()

# Warmup
for _ in range(5):
'''
        
        if kernel_name == 'pytorch_builtin':
            script_content += '    _ = torch.mm(A, B)\n'
        elif kernel_name == 'cuda_basic':
            script_content += '    _ = matmul_cuda.matmul_basic(A, B)\n'
        elif kernel_name == 'cuda_shared':
            script_content += '    _ = matmul_cuda.matmul_shared(A, B)\n'
        elif kernel_name == 'cuda_static_shared':
            script_content += '    _ = matmul_cuda.matmul_static_shared(A, B)\n'
        elif kernel_name.startswith('cuda_template_'):
            tile_width = int(kernel_name.split('_')[-1])
            script_content += f'    _ = matmul_cuda.matmul_template(A, B, {tile_width})\n'
        
        script_content += f'''
torch.cuda.synchronize()

# Profile target
for _ in range({iterations}):
'''
        
        if kernel_name == 'pytorch_builtin':
            script_content += '    result = torch.mm(A, B)\n'
        elif kernel_name == 'cuda_basic':
            script_content += '    result = matmul_cuda.matmul_basic(A, B)\n'
        elif kernel_name == 'cuda_shared':
            script_content += '    result = matmul_cuda.matmul_shared(A, B)\n'
        elif kernel_name == 'cuda_static_shared':
            script_content += '    result = matmul_cuda.matmul_static_shared(A, B)\n'
        elif kernel_name.startswith('cuda_template_'):
            tile_width = int(kernel_name.split('_')[-1])
            script_content += f'    result = matmul_cuda.matmul_template(A, B, {tile_width})\n'
            
        script_content += 'torch.cuda.synchronize()\n'
        
        script_path = self.output_dir / f"profile_{kernel_name}_{M}x{K}x{N}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def profile_kernel(self, kernel_name: str, matrix_size: Tuple[int, int, int], 
                      metrics: List[str], iterations: int = 10, use_sudo: bool = False) -> Optional[str]:
        """Profile a specific kernel with NCU."""
        M, K, N = matrix_size
        
        # Create profile script
        script_path = self.create_profile_script(kernel_name, matrix_size, metrics, iterations)
        
        # Output file
        output_file = self.output_dir / f"ncu_{kernel_name}_{M}x{K}x{N}.ncu-rep"
        
        # Build NCU command
        ncu_cmd = ['ncu']
        if use_sudo:
            ncu_cmd = ['sudo'] + ncu_cmd
            
        ncu_cmd.extend([
            '--metrics', ','.join(metrics),
            '--target-processes', 'all',
            '--export', str(output_file),
            '--force-overwrite',
            'python', script_path
        ])
        
        print(f"Profiling {kernel_name} with matrix size {M}x{K}x{N}...")
        print(f"NCU command: {' '.join(ncu_cmd)}")
        
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"[PASS] Successfully profiled {kernel_name}")
                return str(output_file)
            else:
                print(f"[FAIL] Failed to profile {kernel_name}")
                print(f"Error: {result.stderr}")
                if 'ERR_NVGPUCTRPERM' in result.stderr:
                    print("\n" + "="*60)
                    print("PERMISSION ERROR DETECTED!")
                    print("This error occurs when NCU doesn't have sufficient permissions.")
                    print("\nSolutions:")
                    print("1. Use sudo: sudo python ncu_profiler.py ...")
                    print("2. Set system option: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf")
                    print("3. Use alternative: python profiling_alternative.py ...")
                    print("="*60)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[FAIL] Timeout profiling {kernel_name}")
            return None
        except Exception as e:
            print(f"[FAIL] Error profiling {kernel_name}: {e}")
            return None
    
    def profile_multiple_kernels(self, kernels: List[str], matrix_sizes: List[Tuple[int, int, int]], 
                               metrics_set: str = 'basic', iterations: int = 10, 
                               use_sudo: bool = False) -> Dict[str, List[str]]:
        """Profile multiple kernels with multiple matrix sizes."""
        if not self.load_extension():
            return {}
            
        metrics = self.metrics_sets.get(metrics_set, self.metrics_sets['basic'])
        results = {}
        
        print(f"Profiling {len(kernels)} kernels with {len(matrix_sizes)} matrix sizes")
        print(f"Metrics set: {metrics_set} ({len(metrics)} metrics)")
        print(f"Iterations per kernel: {iterations}")
        print("-" * 60)
        
        for kernel in kernels:
            if kernel not in self.kernels_info:
                print(f"Warning: Unknown kernel {kernel}, skipping...")
                continue
                
            print(f"\n[INFO] Profiling {kernel}: {self.kernels_info[kernel]}")
            kernel_results = []
            
            for matrix_size in matrix_sizes:
                profile_file = self.profile_kernel(kernel, matrix_size, metrics, iterations, use_sudo)
                if profile_file:
                    kernel_results.append(profile_file)
                    
            if kernel_results:
                results[kernel] = kernel_results
                
        return results
    
    def generate_analysis_script(self, profile_files: Dict[str, List[str]]):
        """Generate analysis script for NCU profile files."""
        script_path = self.output_dir / "analyze_ncu_profiles.sh"
        
        script_content = """#!/bin/bash
# NCU Profile Analysis Script
# Generated automatically by ncu_profiler.py

echo "==================================================="
echo "Nsight Compute Profile Analysis"
echo "==================================================="

"""
        
        for kernel, files in profile_files.items():
            script_content += f'\necho "\\n[INFO] Analyzing {kernel}..."\n'
            for profile_file in files:
                profile_name = Path(profile_file).stem
                script_content += f'''
echo "--- {profile_name} ---"
ncu --import {profile_file} --print-summary
echo ""
'''
        
        script_content += '''
echo "\\n[CHART] Generating detailed reports..."
for ncu_file in *.ncu-rep; do
    if [ -f "$ncu_file" ]; then
        base_name=$(basename "$ncu_file" .ncu-rep)
        echo "Exporting $base_name to CSV..."
        ncu --import "$ncu_file" --csv > "${base_name}.csv" 2>/dev/null || echo "Failed to export $base_name"
    fi
done

echo "\\n[INFO] Analysis complete!"
echo "Profile files: *.ncu-rep"
echo "CSV exports: *.csv"
echo "To view in GUI: ncu --import <profile_file>"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"[PASS] Generated analysis script: {script_path}")
        
    def generate_summary_report(self, profile_files: Dict[str, List[str]], metrics_set: str):
        """Generate a summary report."""
        report_path = self.output_dir / "profiling_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# Nsight Compute Profiling Summary\\n\\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Metrics Set:** {metrics_set}\\n")
            f.write(f"**Total Kernels:** {len(profile_files)}\\n\\n")
            
            f.write("## Profile Files\\n\\n")
            for kernel, files in profile_files.items():
                f.write(f"### {kernel}\\n")
                f.write(f"*{self.kernels_info.get(kernel, 'Unknown kernel')}*\\n\\n")
                for profile_file in files:
                    filename = Path(profile_file).name
                    f.write(f"- `{filename}`\\n")
                f.write("\\n")
                
            f.write("## Analysis Commands\\n\\n")
            f.write("```bash\\n")
            f.write("# Run automatic analysis\\n")
            f.write("./analyze_ncu_profiles.sh\\n\\n")
            f.write("# View specific profile in GUI\\n")
            f.write("ncu --import <profile_file>\\n\\n")
            f.write("# Export to CSV\\n")
            f.write("ncu --import <profile_file> --csv > output.csv\\n")
            f.write("```\\n\\n")
            
            f.write("## Key Metrics\\n\\n")
            metrics = self.metrics_sets.get(metrics_set, [])
            for metric in metrics:
                f.write(f"- `{metric}`\\n")
            f.write("\\n")
            
            f.write("## Next Steps\\n\\n")
            f.write("1. Run `./analyze_ncu_profiles.sh` for quick analysis\\n")
            f.write("2. Open profile files in Nsight Compute GUI for detailed view\\n")
            f.write("3. Compare kernel performance using the CSV exports\\n")
            f.write("4. Use profiling_alternative.py for PyTorch Profiler analysis\\n")
            
        print(f"[PASS] Generated summary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Nsight Compute Profiler for CUDA MatMul Kernels')
    
    parser.add_argument('--sizes', type=int, nargs='+', default=[1024],
                       help='Matrix sizes to test (default: 1024)')
    parser.add_argument('--kernels', nargs='+', 
                       default=['cuda_template_16', 'cuda_template_32'],
                       choices=['pytorch_builtin', 'cuda_basic', 'cuda_shared', 
                               'cuda_static_shared', 'cuda_template_8', 
                               'cuda_template_16', 'cuda_template_32'],
                       help='Kernels to profile')
    parser.add_argument('--metrics', default='basic',
                       choices=['basic', 'memory', 'compute', 'occupancy', 'all'],
                       help='Metrics set to collect')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per kernel')
    parser.add_argument('--output', default=None,
                       help='Output directory for profile files')
    parser.add_argument('--sudo', action='store_true',
                       help='Use sudo for NCU (needed for some systems)')
    
    args = parser.parse_args()
    
    # Create matrix sizes (assuming square matrices for simplicity)
    matrix_sizes = [(size, size, size) for size in args.sizes]
    
    # Initialize profiler
    profiler = NCUProfiler(args.output)
    
    # Check GPU compatibility
    gpu_compatible, gpu_msg = profiler.check_gpu_compatibility()
    print(f"GPU 兼容性: {gpu_msg}")
    
    if not gpu_compatible:
        print("\\n[FAIL] 当前 GPU 不支持 Nsight Compute!")
        print("建议使用替代方案:")
        print("  - Nsight Systems: python profile_nsight.py --mode nsys")
        print("  - PyTorch Profiler: python profiling_alternative.py")
        return 1
    
    # Check permissions
    if not args.sudo and not profiler.check_ncu_permissions():
        print("[WARN] NCU permission check failed!")
        print("You may need to use --sudo or configure system permissions.")
        print("Alternatively, use: python profiling_alternative.py")
        
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting. Use --sudo or configure permissions.")
            return 1
    
    print(f"[INFO] Starting NCU profiling...")
    print(f"Output directory: {profiler.output_dir}")
    print(f"Matrix sizes: {args.sizes}")
    print(f"Kernels: {args.kernels}")
    print(f"Metrics: {args.metrics}")
    print(f"Using sudo: {args.sudo}")
    
    # Profile kernels
    profile_files = profiler.profile_multiple_kernels(
        args.kernels, matrix_sizes, args.metrics, args.iterations, args.sudo
    )
    
    if not profile_files:
        print("[FAIL] No successful profiles generated!")
        print("\\nTroubleshooting:")
        print("1. Check CUDA installation: nvcc --version")
        print("2. Try with sudo: python ncu_profiler.py --sudo ...")
        print("3. Use alternative: python profiling_alternative.py")
        return 1
    
    # Generate analysis tools
    profiler.generate_analysis_script(profile_files)
    profiler.generate_summary_report(profile_files, args.metrics)
    
    print(f"\\n[INFO] Profiling complete!")
    print(f"[INFO] Results saved to: {profiler.output_dir}")
    print(f"[INFO] Profile files: {sum(len(files) for files in profile_files.values())}")
    print(f"\\n[INFO] Next steps:")
    print(f"   cd {profiler.output_dir}")
    print(f"   ./analyze_ncu_profiles.sh")
    print(f"   # or open *.ncu-rep files in Nsight Compute GUI")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
