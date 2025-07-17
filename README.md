# Universal Operator Benchmarking Framework

A flexible framework for comparing operator implementations across PyTorch, CUDA, CuPy, and other backends, with specialized high-performance testing for H100/B200 class GPUs.

## Quick Start

### 1. Build the Framework

```bash
./build.sh                          # Build with CUDA extension
./build.sh install-deps             # Auto-install dependencies
./build.sh --skip-cuda              # Framework-only mode
```

### 2. List Available Operators

```bash
python run_comparator.py --list-operators
```

### 3. High-Performance FP8 Testing (H100/B200)

For high-end GPU testing with automatic optimization:

```bash
# Auto-detect GPU and run optimal high-performance tests
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive

# Include stress tests for maximum performance
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive --stress-tests

# Profile GPU capabilities only
python tools/operator_comparator_tool.py --adaptive --profile-only

# Use predefined H100 configuration
python tools/operator_comparator_tool.py --config h100_performance_suite --plot

# B200 stress testing
python tools/operator_comparator_tool.py --config b200_peak_stress --plot
```

### 4. Standard Benchmarks

### 4. Standard Benchmarks

The framework provides two modes:

#### Performance Mode (Default)
Pure performance benchmarking:

```bash
# Basic performance comparison
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square

# With charts
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --plot
```

**Sample Output:**
```
=== Testing MATMUL - small_square ===
  [PERF] PyTorch torch.mm: 0.050ms, 677.7 GFLOPS
  [PERF] PyTorch torch.matmul: 0.048ms, 699.0 GFLOPS
  [PERF] Basic CUDA Kernel: 0.103ms, 324.5 GFLOPS
```

#### Accuracy Mode
Precision verification with baseline reference:

```bash
# Accuracy comparison
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --accuracy-only
```

**Sample Output:**
```
Accuracy Results for small_square:
Baseline (Reference): PyTorch torch.mm (pytorch_mm)
pytorch_mm: PASS (tolerance: 1e-06)
cuda_basic: PASS (tolerance: 0.0001)
cuda_shared: PASS (tolerance: 0.0001)
```

## Supported Operators

### Compute Operators
- **MatMul**: Matrix multiplication (CPU, PyTorch, CUDA implementations)
- **VectorAdd**: Element-wise vector addition
- **ReLU**: Rectified Linear Unit activation
- **RMSNorm**: Root Mean Square Normalization
- **FP8 Linear**: High-performance FP8 linear layers with H100/B200 optimization
  - PyTorch BF16, Transformer Engine, CUTLASS, cuBLAS backends
  - TFLOPS to PFLOPS scale test cases
  - Auto-adaptive GPU detection and optimization
  - Real-world LLM workload scenarios (LLaMA3-405B, GPT-4 scale)

### Network Operators
- **TCP Bandwidth**: TCP network testing
- **RDMA Bandwidth**: RDMA network testing  
- **PCIe Bandwidth**: GPU memory bandwidth testing

## FP8 Linear High-Performance Benchmarking

The FP8 Linear operator provides comprehensive benchmarking optimized for H100/B200 class GPUs with TFLOPS to PFLOPS scale workloads:

### Quick Start with Auto-Adaptive Testing
```bash
# Automatically detect GPU and run optimal tests
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive

# Include stress tests for maximum performance
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive --stress-tests

# Profile GPU capabilities only
python tools/operator_comparator_tool.py --adaptive --profile-only
```

### Predefined Configurations
```bash
# H100 optimized test suite
python tools/operator_comparator_tool.py --config h100_performance_suite --plot

# B200 stress testing
python tools/operator_comparator_tool.py --config b200_peak_stress --plot

# Quick validation
python tools/operator_comparator_tool.py --config h100_quick_validation --plot
```

### Manual High-Performance Testing
```bash
# H100 optimized test suite
python tools/operator_comparator_tool.py --operator fp8_linear \
  --test-cases h100_target_small h100_target_medium h100_target_large \
  --plot --runs 5

# B200 stress testing
python tools/operator_comparator_tool.py --operator fp8_linear \
  --test-cases b200_peak_stress --runs 3 --plot

# Real-world LLM workloads
python tools/operator_comparator_tool.py --operator fp8_linear \
  --test-cases llama3_405b_ffn_up llama3_405b_ffn_down gpt4_scale_attention \
  --plot --runs 3
```

### Performance Test Cases

**H100 Optimized (1000 TFLOPS FP8 peak):**
- `h100_target_small`: 8K×8K→8K (1.07 TFLOPS)
- `h100_target_medium`: 16K×12K→16K (6.44 TFLOPS)  
- `h100_target_large`: 32K×16K→32K (35.18 TFLOPS)
- `h100_peak_stress`: 64K×24K→48K (157.29 TFLOPS)

**B200 Optimized (2500 TFLOPS FP8 peak):**
- `b200_target_medium`: 64K×32K→64K (274.88 TFLOPS)
- `b200_target_large`: 128K×48K→96K (1.23 PFLOPS)
- `b200_peak_stress`: 256K×64K→128K (4.40 PFLOPS)

**Real-World LLM Workloads:**
- `llama3_405b_ffn_up/down`: LLaMA3-405B scale FFN projections
- `gpt4_scale_attention`: GPT-4 scale attention (824+ TFLOPS)

### Configuration Management
```bash
# List available predefined configurations
python tools/operator_comparator_tool.py --config --help

# Use H100 performance suite
python tools/operator_comparator_tool.py --config h100_performance_suite --plot

# Use B200 peak stress configuration
python tools/operator_comparator_tool.py --config b200_peak_stress --plot
```

**Available Backends:**
- PyTorch BF16 (baseline)
- Simulated FP8 Fast (high-performance)
- Simulated FP8 Memory-Optimized
- Transformer Engine (optional)
- CUTLASS/cuBLAS (optional)

**Features:**
- CHART Automatic chart generation
- TARGET H100/B200 optimized test cases (1.07 TFLOPS to 4.40 PFLOPS)
- CHART Performance analysis and efficiency metrics
- Auto-adaptive GPU detection and configuration
- Memory bandwidth optimization tests

## Network Testing

```bash
python run_network_tests.py --all      # All network tests
python run_network_tests.py --tcp      # TCP only
python run_network_tests.py --rdma     # RDMA only
```

## Command Line Options

```bash
# List operators and implementations
python tools/operator_comparator_tool.py --list-operators
python tools/operator_comparator_tool.py --operator matmul --list-implementations

# Performance benchmarking
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --warmup 10 --runs 50

# Accuracy comparison
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --accuracy-only

# Custom output
python tools/operator_comparator_tool.py --operator matmul --test-cases small_square --output-dir results --plot
```

## Requirements

- **System**: Linux, GCC with C++17, CMake 3.10+
- **Python**: 3.8+ with PyTorch, NumPy
- **Optional**: Matplotlib (charts), RDMA tools (network testing)

Use `./build.sh install-deps` for automatic dependency installation.

## Profiling

```bash
./scripts/run_profiling.sh           # Nsight Systems
./scripts/run_ncu_profiling.sh       # Nsight Compute
python tools/profiling_pytorch.py    # PyTorch Profiler
```

## Contributing

To add a new operator:

1. Create operator class in `src/operators/` inheriting from `BaseOperator`
2. Implement required methods and test cases
3. Register the operator in the framework

See existing operators for examples.

## Troubleshooting

### Build Issues
```bash
./build.sh check-deps               # Check dependencies
./build.sh install-deps             # Auto-install missing deps
./build.sh --skip-cuda              # Skip CUDA if build fails
```

### Network Testing
- **RDMA tools not found**: Install InfiniBand utilities or skip RDMA tests
- **Permission issues**: Some network tests may require sudo privileges

## License

Apache 2.0 License - see LICENSE file for details.
