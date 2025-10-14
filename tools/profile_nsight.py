#!/usr/bin/env python3
"""
Profile CUDA MatMul operators with Nsight Systems
"""

import torch
import time
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Configure the PyTorch library path to resolve CUDA extension imports
def setup_library_path():
    """Configure library paths so CUDA extensions load correctly"""
    torch_lib_path = os.path.join(torch.__path__[0], 'lib')
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    if torch_lib_path not in current_ld_path:
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = torch_lib_path
        print(f"[INFO] Configured library path: {torch_lib_path}")

# Configure library path before importing other modules
setup_library_path()

class NsightProfiler:
    def __init__(self, output_dir: str = "nsight_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matmul_cuda = None
        
        # Lazy-load the CUDA extension
        self._load_cuda_extension()
    
    def _load_cuda_extension(self):
        """Load the CUDA extension"""
        try:
            import matmul_cuda_ext
            self.matmul_cuda = matmul_cuda_ext
            print("[PASS] CUDA extension loaded successfully")
        except Exception as e:
            print(f"[FAIL] Failed to load CUDA extension: {e}")
            self.matmul_cuda = None
    
    def check_nsight_availability(self) -> bool:
        """Check whether Nsight Systems is available"""
        try:
            result = subprocess.run(['nsys', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[PASS] Nsight Systems available: {result.stdout.strip()}")
                return True
            else:
                print("[FAIL] Nsight Systems unavailable")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[FAIL] nsys command not found; ensure Nsight Systems is installed and on PATH")
            return False
    
    def check_nsight_compute_availability(self) -> bool:
        """Check whether Nsight Compute is available"""
        try:
            result = subprocess.run(['ncu', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"[PASS] Nsight Compute available: {result.stdout.strip().split()[0:3]}")
                return True
            else:
                print("[FAIL] Nsight Compute unavailable")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[FAIL] ncu command not found; ensure Nsight Compute is installed and on PATH")
            return False
    
    def create_test_matrices(self, m: int, n: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test matrices"""
        torch.manual_seed(42)
        A = torch.randn(m, k, device=self.device, dtype=torch.float32).contiguous()
        B = torch.randn(k, n, device=self.device, dtype=torch.float32).contiguous()
        return A, B
    
    def warmup_kernels(self, A: torch.Tensor, B: torch.Tensor, implementations: Dict):
        """Warm up all kernels"""
        print("Warming up kernels...")
        for name, func in implementations.items():
            try:
                for _ in range(3):
                    _ = func(A, B)
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Failed to warm up {name}: {e}")
    
    def profile_single_kernel(self, kernel_name: str, kernel_func, A: torch.Tensor, B: torch.Tensor, 
                             matrix_size: str, iterations: int = 100) -> str:
        """Profile a single kernel"""
        
        # Generate a temporary Python script for profiling
        temp_script = self.output_dir / f"temp_profile_{kernel_name}.py"
        output_file = self.output_dir / f"{kernel_name}_{matrix_size}_profile.nsys-rep"
        
        # Write temporary script
        script_content = f'''
import sys
import os
# Add current directory to sys.path to locate compiled extensions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "{os.getcwd()}")

import torch
import matmul_cuda_ext
import time

# Recreate identical matrices
torch.manual_seed(42)
A = torch.randn({A.shape[0]}, {A.shape[1]}, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn({B.shape[0]}, {B.shape[1]}, device='cuda', dtype=torch.float32).contiguous()

# Select the matching function
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

# Warmup
for _ in range(5):
    _ = func(A, B)
    torch.cuda.synchronize()

# Start profiling region
torch.cuda.nvtx.range_push("MatMul_Profiling_{kernel_name}")

# Run multiple iterations
for i in range({iterations}):
    torch.cuda.nvtx.range_push(f"Iteration_{{i}}")
    result = func(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_pop()
'''
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Build nsys command (compatible with older GPUs; removes unsupported options)
        nsys_cmd = [
            'nsys', 'profile',
            '--output', str(output_file),
            '--force-overwrite', 'true',
            '--trace', 'cuda,nvtx',
            '--cuda-memory-usage', 'true',
            '--duration', '30',  # Cap profiling at 30 seconds
            'python', str(temp_script)
        ]
        
        print(f"  Profiling {kernel_name}...")
        try:
            result = subprocess.run(nsys_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"    [PASS] Profile saved to: {output_file}")
                # Remove temporary file
                temp_script.unlink()
                return str(output_file)
            else:
                print(f"    [FAIL] Profiling failed: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] Profiling timed out")
            return ""
        except Exception as e:
            print(f"    [FAIL] Profiling exception: {e}")
            return ""
    
    def profile_kernel_with_ncu(self, kernel_name: str, A: torch.Tensor, B: torch.Tensor,
                               matrix_size: str, metrics: List[str] = None) -> str:
        """Profile a single kernel with Nsight Compute"""
        
        if metrics is None:
            # Default performance metrics
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
        
        # Generate temporary Python script for NCU profiling
        temp_script = self.output_dir / f"temp_ncu_{kernel_name}.py"
        output_file = self.output_dir / f"{kernel_name}_{matrix_size}_ncu_profile.ncu-rep"
        
        # Write temporary script
        script_content = f'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "{os.getcwd()}")

import torch
import matmul_cuda_ext

# Recreate identical matrices
torch.manual_seed(42)
A = torch.randn({A.shape[0]}, {A.shape[1]}, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn({B.shape[0]}, {B.shape[1]}, device='cuda', dtype=torch.float32).contiguous()

# Select the matching function
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

# Warmup
for _ in range(3):
    _ = func(A, B)
    torch.cuda.synchronize()

# Run the kernel for profiling
result = func(A, B)
torch.cuda.synchronize()
'''
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Build ncu command
        metrics_args = ['--metrics', ','.join(metrics)]  # Join metrics with commas
        
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
                print(f"    [PASS] NCU Profile saved to: {output_file}")
                # Remove temporary file
                temp_script.unlink()
                return str(output_file)
            else:
                print(f"    [FAIL] NCU Profiling failed: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            print(f"    [FAIL] NCU Profiling timed out")
            return ""
        except Exception as e:
            print(f"    [FAIL] NCU Profiling exception: {e}")
            return ""

    def profile_all_kernels(self, matrix_sizes: List[Tuple[int, int, int]], 
                           iterations: int = 100) -> Dict[str, List[str]]:
        """Profile all kernels"""
        
        if not self.check_nsight_availability():
            return {}
        
        if not self.matmul_cuda:
            print("CUDA extension not loaded; cannot profile")
            return {}
        
        # Define available implementations
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
            print(f"\n=== Profiling matrix dimensions: ({m}, {k}) × ({k}, {n}) ===")
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
                
                # Short pause to avoid GPU overheating
                time.sleep(1)
        
        return profile_files
    
    def profile_all_kernels_ncu(self, matrix_sizes: List[Tuple[int, int, int]], 
                               kernel_names: List[str] = None) -> Dict[str, List[str]]:
        """Profile specified kernel with Nsight Compute"""
        
        if not self.check_nsight_compute_availability():
            return {}
        
        if not self.matmul_cuda:
            print("CUDA extension not loaded; cannot perform NCU profiling")
            return {}
        
        if kernel_names is None:
            # Default to profiling several key kernels
            kernel_names = ['pytorch_builtin', 'cuda_basic', 'cuda_shared', 'cuda_template_32']
        
        profile_files = {}
        
        for m, k, n in matrix_sizes:
            print(f"\n=== NCU Profiling matrix dimensions: ({m}, {k}) × ({k}, {n}) ===")
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
                
                # Rest briefly to avoid GPU overheating
                time.sleep(2)
        
        return profile_files

    def generate_analysis_script(self, profile_files: Dict[str, List[str]]):
        """Generate script to analyze profile files"""
        analysis_script = self.output_dir / "analyze_profiles.sh"
        
        script_content = '''#!/bin/bash
# Nsight Systems profile analysis script
# Usage: ./analyze_profiles.sh

echo "=== Nsight Systems Profile Analysis ==="
echo

# Verify nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys command not found"
    exit 1
fi

'''
        
        # Append analysis commands for each profile file
        for kernel_name, files in profile_files.items():
            script_content += f'\necho "=== Analyzing {kernel_name} ==="\n'
            for profile_file in files:
                if os.path.exists(profile_file):
                    profile_basename = os.path.basename(profile_file)
                    script_content += f'''
echo "Profile file: {profile_basename}"
echo "Generate statistical report..."
nsys stats --report gputrace,gpukernsum,gpumemtimesum {profile_basename}
echo
'''
        
        script_content += r'''
echo "=== Generate visualization report ==="
echo "Open .nsys-rep files in Nsight Systems GUI for detailed analysis"
echo "Or use the following command to produce a SQLite database:"
echo

for file in *.nsys-rep; do
    if [ -f "$file" ]; then
        echo "nsys export --type sqlite --output ${file%.nsys-rep}.sqlite $file"
    fi
done
'''
        
        with open(analysis_script, 'w') as f:
            f.write(script_content)
        
    # Make script executable
        os.chmod(analysis_script, 0o755)
        print(f"\n[PASS] Analysis script generated: {analysis_script}")
    
    def generate_ncu_analysis_script(self, ncu_profile_files: Dict[str, List[str]]):
        """Generate Nsight Compute analysis script"""
        analysis_script = self.output_dir / "analyze_ncu_profiles.sh"
        
        script_content = '''#!/bin/bash
# Nsight Compute profile analysis script
# Usage: ./analyze_ncu_profiles.sh

echo "=== Nsight Compute Profile Analysis ==="
echo

# Verify ncu is available
if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu command not found"
    exit 1
fi

'''
        
        # Append analysis commands for each NCU profile file
        for kernel_name, files in ncu_profile_files.items():
            script_content += f'\necho "=== NCU analysis {kernel_name} ==="\\n'
            for profile_file in files:
                if os.path.exists(profile_file):
                    profile_basename = os.path.basename(profile_file)
                    script_content += f'''
echo "NCU Profile file: {profile_basename}"
echo "Generate detailed report..."
ncu --import {profile_basename} --print-summary
echo
echo "Generate CSV report..."
ncu --import {profile_basename} --csv > "${{profile_basename%.ncu-rep}}_summary.csv"
echo
'''
        
        script_content += r'''
echo "=== Generate comparison report ==="
echo "Open .ncu-rep files in Nsight Compute GUI for detailed inspection"
echo "Or use the following command to export detailed data:"
echo

for file in *.ncu-rep; do
    if [ -f "$file" ]; then
        echo "ncu --import $file --csv > ${file%.ncu-rep}_detailed.csv"
    fi
done
'''
        
        with open(analysis_script, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(analysis_script, 0o755)
        print(f"\n[PASS] NCU Analysis script generated: {analysis_script}")
    
    def generate_summary_report(self, profile_files: Dict[str, List[str]]):
        """Generate profiling summary report."""
        report_file = self.output_dir / "profiling_summary.md"

        content = f"""# CUDA MatMul Nsight Systems profiling report

## Environment
- Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch version: {torch.__version__}
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Profile files

"""

        for kernel_name, files in profile_files.items():
            content += f"### {kernel_name}\n\n"
            for profile_file in files:
                if os.path.exists(profile_file):
                    file_size = os.path.getsize(profile_file) / 1024 / 1024  # MB
                    content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"
            content += "\n"

        content += """## Usage

### 1. Command line analysis
```bash
# Run the analysis script
./analyze_profiles.sh

# Or manually analyze an individual file
nsys stats --report gputrace,gpukernsum,gpumemtimesum <profile_file.nsys-rep>
```

### 2. GUI analysis
```bash
# Launch Nsight Systems GUI
nsight-sys

# Then open the .nsys-rep file for visualization
```

### 3. Export data
```bash
# Export to SQLite database
nsys export --type sqlite --output profile_data.sqlite <profile_file.nsys-rep>
```

## Key metrics

1. **Kernel execution time**: Actual GPU kernel runtimes
2. **Memory transfers**: Host-device transfer durations
3. **GPU utilization**: Utilization of GPU compute units
4. **Memory bandwidth**: Efficiency of memory accesses
5. **Thread occupancy**: Streaming multiprocessor occupancy

## Optimization suggestions

Focus on these aspects when interpreting the profiling results:
- Kernel launch overhead
- Memory access patterns
- Thread block size
- Shared memory efficiency
- Register usage
"""

        with open(report_file, "w") as f:
            f.write(content)

        print(f"[PASS] Summary report generated: {report_file}")

    def generate_comprehensive_report(self, nsys_files: Dict[str, List[str]],
                                      ncu_files: Dict[str, List[str]], mode: str):
        """Generate a combined Nsight Systems and Nsight Compute report."""
        report_file = self.output_dir / "comprehensive_profiling_report.md"

        content = f"""# CUDA MatMul comprehensive performance report

## Environment
- Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- PyTorch version: {torch.__version__}
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Profiling mode: {mode}

## Comparison of analysis tools

### Nsight Systems vs Nsight Compute

| Tool | Purpose | Analysis granularity | Key metrics |
|------|---------|----------------------|-------------|
| Nsight Systems | System-level performance analysis | Whole application | Timelines, GPU utilization, memory transfers |
| Nsight Compute | Kernel-level detailed analysis | Individual CUDA kernel | Instruction throughput, memory efficiency, occupancy |

"""

        if nsys_files:
            content += """## Nsight Systems findings

### Timeline analysis
Nsight Systems provides a full execution timeline to inspect:
- Kernel launch overhead
- Memory transfer timing
- CPU-GPU synchronization points
- Overall program flow

### Profile files
"""
            for kernel_name, files in nsys_files.items():
                content += f"\n#### {kernel_name}\n"
                for profile_file in files:
                    if os.path.exists(profile_file):
                        file_size = os.path.getsize(profile_file) / 1024 / 1024
                        content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"

        if ncu_files:
            content += """\n## Nsight Compute findings

### Detailed kernel analysis
Nsight Compute provides per-kernel metrics covering:
- Instruction-level throughput
- Memory subsystem efficiency
- Register and shared-memory consumption
- Thread occupancy and latency

### Profile files
"""
            for kernel_name, files in ncu_files.items():
                content += f"\n#### {kernel_name}\n"
                for profile_file in files:
                    if os.path.exists(profile_file):
                        file_size = os.path.getsize(profile_file) / 1024 / 1024
                        content += f"- `{os.path.basename(profile_file)}` ({file_size:.1f} MB)\n"

        content += """\n## Analysis workflow

### 1. System-level analysis (Nsight Systems)
```bash
# Run Nsight Systems analysis
./analyze_profiles.sh

# Or analyze manually
nsys stats --report gputrace,gpukernsum profile.nsys-rep
```

### 2. Kernel-level analysis (Nsight Compute)
```bash
# Run Nsight Compute analysis
./analyze_ncu_profiles.sh

# Or analyze manually
ncu --import profile.ncu-rep --print-summary
```

### 3. GUI visualization
```bash
# Nsight Systems GUI
nsight-sys *.nsys-rep

# Nsight Compute GUI
ncu-ui *.ncu-rep
```

## Key performance indicators

### Nsight Systems metrics
1. **GPU utilization**: Overall GPU usage
2. **Memory bandwidth**: DRAM throughput
3. **Kernel execution time**: Per-kernel runtime comparison
4. **Launch overhead**: Kernel launch and scheduling cost

### Nsight Compute metrics
1. **Compute throughput**:
    - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
    - Efficiency of floating-point instruction issue

2. **Memory efficiency**:
    - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
    - `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`

3. **Occupancy**:
    - `sm__warps_active.avg.pct_of_peak_sustained_active`
    - Register and shared-memory utilization

## Optimization suggestions

### Insights from Nsight Systems
1. **Reduce kernel launch overhead**: Fuse small kernel invocations when possible
2. **Optimize memory transfers**: Minimize host-device data movement
3. **Improve GPU utilization**: Keep the GPU consistently fed with work

### Insights from Nsight Compute
1. **Increase compute throughput**:
    - Tune instruction mix and expose parallelism
    - Reduce branch divergence

2. **Improve memory access**:
    - Promote memory coalescing
    - Increase cache hit rates
    - Leverage shared memory effectively

3. **Boost occupancy**:
    - Adjust thread-block sizes
    - Optimize register usage
    - Balance shared-memory allocation

## Tool tips

### Nsight Systems
- Use NVTX ranges to flag critical regions
- Compare scaling trends across matrix sizes
- Watch for idle periods and synchronization gaps

### Nsight Compute
- Identify compute- versus memory-bound behavior
- Apply roofline analysis to gauge headroom
- Compare kernels after each optimization pass

## Next steps

1. **Run the analysis scripts**: Gather quantitative metrics
2. **Inspect with the GUIs**: Explore performance bottlenecks visually
3. **Compare implementations**: Evaluate key metrics across kernels
4. **Iterate on optimizations**: Apply changes based on profiler feedback
"""

        with open(report_file, "w") as f:
            f.write(content)

    print(f"[PASS] Comprehensive analysis report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='CUDA MatMul Nsight Systems/Compute Profiling')
    parser.add_argument('--output', '-o', default='nsight_profiles', 
                       help='Output directory (default: nsight_profiles)')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Iterations per kernel (default: 100)')
    parser.add_argument('--sizes', nargs='+', default=['1024'],
                       help='Matrix size list (default: [1024])')
    parser.add_argument('--kernels', nargs='+', 
                       default=['pytorch_builtin', 'cuda_template_32', 'cuda_static_shared'],
                       help='Kernels to profile')
    parser.add_argument('--mode', choices=['nsys', 'ncu', 'both'], default='nsys',
                       help='Profiling mode: nsys (Nsight Systems), ncu (Nsight Compute), both (default: nsys)')
    
    # Parameters consistent with README documentation
    parser.add_argument('--operator', default='matmul',
                       help='Operator to analyze (default: matmul)')
    parser.add_argument('--test-case', default='large_square',
                       help='Test case name (default: large_square)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Profiling duration (seconds) (default: 10)')
    
    args = parser.parse_args()
    
    # Parse matrix sizes
    matrix_sizes = []
    for size_str in args.sizes:
        try:
            size = int(size_str)
            matrix_sizes.append((size, size, size))  # square matrix
        except ValueError:
            print(f"Invalid matrix size: {size_str}")
            continue
    
    if not matrix_sizes:
        matrix_sizes = [(1024, 1024, 1024)]  # default
    
    print("CUDA MatMul Nsight Profiler")
    print("=" * 50)
    print(f"Operator: {args.operator}")
    print(f"Test case: {args.test_case}")
    print(f"Output directory: {args.output}")
    print(f"Iterations: {args.iterations}")
    print(f"Duration: {args.duration}s")
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Profile kernels: {args.kernels}")
    print(f"Profiling mode: {args.mode}")
    print()
    
    profiler = NsightProfiler(args.output)
    
    nsys_profile_files = {}
    ncu_profile_files = {}
    
    # Run Nsight Systems profiling
    if args.mode in ['nsys', 'both']:
        print("=== Run Nsight Systems profiling ===")
        nsys_profile_files = profiler.profile_all_kernels(matrix_sizes, args.iterations)
        
        if nsys_profile_files:
            profiler.generate_analysis_script(nsys_profile_files)
    
    # Run Nsight Compute profiling
    if args.mode in ['ncu', 'both']:
        print("\n=== Run Nsight Compute profiling ===")
        ncu_profile_files = profiler.profile_all_kernels_ncu(matrix_sizes, args.kernels)
        
        if ncu_profile_files:
            profiler.generate_ncu_analysis_script(ncu_profile_files)
    
    # Generate combined report
    if nsys_profile_files or ncu_profile_files:
        profiler.generate_comprehensive_report(nsys_profile_files, ncu_profile_files, args.mode)
        
        print("\n" + "=" * 50)
        print("Profiling complete!")
        print(f"Results saved to: {args.output}/")
        print("Next steps:")
        
        if nsys_profile_files:
            print(f"  1. Nsight Systems: cd {args.output} && ./analyze_profiles.sh")
        if ncu_profile_files:
            print(f"  2. Nsight Compute: cd {args.output} && ./analyze_ncu_profiles.sh")
        
        print(f"  3. Or open the profile files in the GUI for detailed analysis")
    else:
        print("Profiling failed, please check environment configuration")

if __name__ == "__main__":
    main()
