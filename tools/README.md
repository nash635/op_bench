# Operator Framework Tools

This directory contains tools for comparing and profiling operators within the framework.

## Core Tools

### 1. Universal Operator Comparator ⭐ (主要工具)
- **File**: `operator_comparator_tool.py`
- **Functionality**: 主要的算子对比工具，支持两种模式：
  - **性能模式** (默认): 纯性能测试，显示 `[PERF]` 状态
  - **精度模式** (`--accuracy-only`): 专门的精度验证，显示基准参考和误差分析
- **Usage**: 用于性能分析、精度验证和生成对比图表
- **Examples**:
  ```bash
  # 性能模式 (默认)
  python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --plot
  
  # 精度模式
  python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --accuracy-only
  ```

### 2. Nsight Compute Profiler
- **File**: `ncu_profiler.py` 
- **Functionality**: 使用 NVIDIA Nsight Compute 进行详细的 CUDA kernel 性能分析
- **Usage**: 深度内核优化和性能瓶颈识别
- **Example**:
  ```bash
  python tools/ncu_profiler.py --sizes 1024 --kernels cuda_template_16
  ```

### 3. Comprehensive Nsight Profiler
- **File**: `profile_nsight.py`
- **Functionality**: 同时支持 Nsight Systems 和 Nsight Compute 的综合分析工具
- **Usage**: 系统级性能分析和应用时间线分析
- **Example**:
  ```bash
  python tools/profile_nsight.py --operator matmul --test-case large_square
  ```

### 4. PyTorch Profiler (权限友好)
- **File**: `profiling_pytorch.py`
- **Functionality**: 当 Nsight 工具权限不足时的替代分析方案
- **Usage**: 无需特殊权限的性能分析
- **Example**:
  ```bash
  python tools/profiling_pytorch.py --sizes 1024 --output results
  ```

## Usage Scenarios

### Performance Analysis (性能模式)
Use the operator comparator tool to compare different implementations focused on performance:
```bash
# 性能对比所有 MatMul 实现并生成图表
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --plot

# 对比特定的 ReLU 测试用例性能
python tools/operator_comparator_tool.py --operator relu --test-cases small_tensor large_tensor

# 多个测试用例的性能分析
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square medium_square large_square --plot
```

### Accuracy Analysis (精度模式)
Use the accuracy-only mode for precision verification:
```bash
# 基础精度对比，显示基准参考
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --accuracy-only

# 特定实现的精度验证
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --implementations pytorch_mm cuda_basic cuda_shared --accuracy-only

# 多测试用例精度分析
python tools/operator_comparator_tool.py --operator vector_add --test-cases small_vectors medium_vectors --accuracy-only
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
