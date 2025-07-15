# Universal Operator Benchmarking Framework

A flexible framework for comparing operator implementations across PyTorch, CUDA, CuPy, and other backends.

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

### 3. Run Benchmarks

The framework provides two modes:

#### Performance Mode (Default)
Pure performance benchmarking:

```bash
# Basic performance comparison
python run_comparator.py --operator matmul --test-cases small_square

# With charts
python run_comparator.py --operator matmul --test-cases small_square --plot
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
python run_comparator.py --operator matmul --test-cases small_square --accuracy-only
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

### Network Operators
- **TCP Bandwidth**: TCP network testing
- **RDMA Bandwidth**: RDMA network testing  
- **PCIe Bandwidth**: GPU memory bandwidth testing

## Network Testing

```bash
python run_network_tests.py --all      # All network tests
python run_network_tests.py --tcp      # TCP only
python run_network_tests.py --rdma     # RDMA only
```

## Command Line Options

```bash
# List operators and implementations
python run_comparator.py --list-operators
python run_comparator.py --operator matmul --list-implementations

# Performance benchmarking
python run_comparator.py --operator matmul --test-cases small_square
python run_comparator.py --operator matmul --test-cases small_square --warmup 10 --runs 50

# Accuracy comparison
python run_comparator.py --operator matmul --test-cases small_square --accuracy-only

# Custom output
python run_comparator.py --operator matmul --test-cases small_square --output-dir results --plot
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
