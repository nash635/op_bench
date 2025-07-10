# Operator Framework Tools

This directory contains tools for comparing and profiling operators within the framework.

## Core Tools

### 1. Operator Comparator
- **File**: `operator_comparator_tool.py`
- **Functionality**: A command-line tool to run benchmarks and compare the performance and output of different operator implementations.
- **Usage**: Used for performance analysis, accuracy verification, and generating comparison plots.
- **Example**:
  ```bash
  python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --plot
  ```

### 2. Nsight Compute Profiler
- **File**: `ncu_profiler.py`
- **Functionality**: A Python script to automate profiling of CUDA kernels using NVIDIA's Nsight Compute (`ncu`). It provides detailed performance metrics for individual kernels.
- **Usage**: Ideal for in-depth kernel optimization and identifying performance bottlenecks.
- **Example**:
  ```bash
  python tools/ncu_profiler.py --operator matmul --test-case "matmul_1024x1024x1024"
  ```

### 3. Nsight Systems Profiler
- **File**: `profile_nsight.py`
- **Functionality**: A Python script to automate system-wide performance analysis using NVIDIA's Nsight Systems (`nsys`). It captures interactions between CPU, GPU, and system memory.
- **Usage**: Best for understanding the overall application timeline and identifying system-level issues.
- **Example**:
  ```bash
  python tools/profile_nsight.py --operator matmul --test-case "matmul_1024x1024x1024"
  ```

### 4. PyTorch Profiler
- **File**: `profiling_pytorch.py`
- **Functionality**: Provides PyTorch-based profiling method for CUDA kernels when Nsight Compute is unavailable or encounters permission errors (`ERR_NVGPUCTRPERM`). It uses PyTorch CUDA events to measure kernel execution time.
- **Usage**: A fallback for basic performance measurement without requiring special permissions.
- **Example**:
  ```bash
  python tools/profiling_pytorch.py
  ```

## Usage Scenarios

### Performance Analysis
Use the operator comparator tool to compare different implementations of the same operator:
```bash
# Compare all MatMul implementations and generate plots
python tools/operator_comparator_tool.py --operator matmul --test-cases all --plot --output-dir results

# Compare specific test cases for ReLU
python tools/operator_comparator_tool.py --operator relu --test-cases small_tensor,large_tensor
```

### Detailed Kernel Profiling
When you need detailed kernel-level performance metrics:
```bash
# Profile a specific operator implementation
python tools/ncu_profiler.py --operator matmul --test-case "matmul_1024x1024x1024"

# Profile with custom output directory
python tools/ncu_profiler.py --operator vector_add --test-case small_tensor --output-dir profiling_results
```

### System-wide Performance Analysis
For understanding CPU-GPU interactions and memory transfers:
```bash
# Full system timeline analysis
python tools/profile_nsight.py --operator rmsnorm --test-case small_sequence

# Profile with custom duration
python tools/profile_nsight.py --operator matmul --test-case large_square --duration 30
```

### Permission-limited Environments
When Nsight tools require elevated permissions:
```bash
# Basic timing analysis without special permissions using PyTorch CUDA events
python tools/profiling_pytorch.py

# Check permissions and get guidance
python tools/profiling_pytorch.py --check-permissions
```

## Requirements

### Basic Requirements
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability

### For Nsight Profiling
- NVIDIA Nsight Compute (`ncu`) - for kernel-level profiling
- NVIDIA Nsight Systems (`nsys`) - for system-level profiling
- Appropriate permissions (may require `sudo` or user group membership)

### For PyTorch Profiling
- PyTorch with CUDA support
- PyTorch CUDA events support
- Basic CUDA runtime

## Output Files

The profiling tools generate various output files:

### Nsight Compute (`ncu_profiler.py`)
- `.ncu-rep` files: Detailed kernel metrics (can be opened in Nsight Compute GUI)
- Text reports with key performance indicators
- Kernel execution time breakdowns

### Nsight Systems (`profile_nsight.py`)
- `.nsys-rep` files: System timeline data (can be opened in Nsight Systems GUI)
- Summary reports with CPU/GPU utilization
- Memory transfer analysis

### PyTorch Profiler (`profiling_pytorch.py`)
- Basic timing measurements using PyTorch CUDA events
- Permission check results and guidance
- Simple performance comparisons
- Compatible with any PyTorch CUDA environment

## Troubleshooting

### Permission Errors
If you encounter `ERR_NVGPUCTRPERM` errors:
1. Use the PyTorch profiler: `python tools/profiling_pytorch.py`
2. Run with elevated permissions: `sudo python tools/ncu_profiler.py ...`
3. Add user to appropriate groups (system-dependent)

### Missing Nsight Tools
If `ncu` or `nsys` are not found:
1. Install NVIDIA Nsight Compute and Nsight Systems
2. Ensure they are in your PATH
3. Use the PyTorch profiler as a fallback: `python tools/profiling_pytorch.py`

### CUDA Not Available
If CUDA is not available:
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Some tools will fall back to CPU-only mode
