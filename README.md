# Universal Operator Benchmarking Framework

A flexible, extensible framework for implementing and comparing different operator implementations (perf/output...) across various backends (PyTorch, CUDA, CuPy, etc.).


## Quick Start

### 1. Build the Framework

```bash
# Simple build (attempts CUDA extension - recommended)
./build.sh

# Check and install missing dependencies automatically
./build.sh install-deps             # Install dependencies and exit
./build.sh check-deps               # Check dependencies only

# Build with advanced options
./build.sh --test                   # Build and run tests
./build.sh --benchmark              # Build and run benchmarks
./build.sh --verbose                # Verbose output
./build.sh --debug                  # Show detailed build information

# Alternative build methods
./build.sh --skip-cuda              # Framework-only mode (if CUDA build fails)
./build.sh cuda                     # Force CUDA extension build

# Utility commands
./build.sh clean                    # Clean build artifacts
./build.sh clean-all               # Clean build artifacts and all output (including custom --output-dir results)
./build.sh test-framework           # Test framework functionality
```

**Dependency Management**: The framework now includes automatic dependency detection and installation:
- `./build.sh install-deps` - Automatically detects and installs missing system packages and Python libraries
- Supports multiple Linux distributions (Ubuntu, CentOS, RHEL, Alibaba Cloud Linux, etc.)
- Installs network testing tools (iperf3, RDMA tools, etc.) for comprehensive network benchmarking
- Handles both system packages (via yum/dnf/apt-get) and Python packages (via conda/pip)

### 2. List Available Operators

```bash
python run_comparator.py --list-operators
```

### 3. Run Benchmarks

```bash
# Compare MatMul implementations
python run_comparator.py --operator matmul --test-cases small_square

# Compare VectorAdd implementations
python run_comparator.py --operator vector_add --test-cases small_tensor

# Compare ReLU implementations
python run_comparator.py --operator relu --test-cases small_tensor

# Compare RMSNorm implementations (fused vs unfused)
python run_comparator.py --operator rmsnorm --test-cases small_sequence

# Generate plots and save to a specific directory
python run_comparator.py --operator matmul --test-cases small_square --output-dir my_results --plot

# Compare accuracy differences between implementations (using PyTorch CPU as reference)
python run_comparator.py --operator matmul --test-cases small_square --output-dir result --plot --output-diff
```

### 4. Network Performance Testing

The framework now includes comprehensive network performance testing capabilities:

**Note**: Network tests can take longer than compute tests (2-30 seconds per test case depending on configuration). For quicker testing, use `--warmup 0 --runs 1` parameters.

```bash
# Run all network tests
python run_network_tests.py --all

# Run specific network tests
python run_network_tests.py --tcp          # TCP bandwidth tests
python run_network_tests.py --rdma         # RDMA bandwidth tests
python run_network_tests.py --pcie         # PCIe bandwidth tests
python run_network_tests.py --stress       # Network stress tests

# Test individual network operators (note: network tests may take some time)
python run_comparator.py --operator tcp_bandwidth --test-cases tcp_bandwidth_64KB --warmup 1 --runs 3
python run_comparator.py --operator pcie_bandwidth --test-cases pcie_bandwidth_1MB --warmup 1 --runs 3

# For quicker testing, use fewer runs and smaller buffers
python run_comparator.py --operator tcp_bandwidth --test-cases tcp_bandwidth_64KB --warmup 0 --runs 1
python run_comparator.py --operator pcie_bandwidth --test-cases pcie_bandwidth_1MB --warmup 0 --runs 1
```

## Framework Features

### Modular Design
- **BaseOperator**: Abstract base class for all operators
- **OperatorType**: Enum defining supported operator types
- **OperatorTestCase**: Configuration for test cases
- **ImplementationResult**: Structured result format

### Extensible Architecture
- Easy to add new operators
- Support for multiple implementations per operator
- Flexible test case configuration
- Comprehensive benchmarking and visualization

### Performance Analysis
- Execution time measurements
- Memory usage tracking
- GFLOPS calculations
- Comparative visualizations
- JSON and Markdown reporting

### Accuracy Analysis
- Precision difference comparison between implementations
- PyTorch CPU implementation as reference baseline
- Statistical error metrics (MSE, MAE, relative error)
- Relative error and absolute error bar charts

## Output Examples

The framework generates:
- **Performance charts** (PNG): Execution time and GFLOPS comparisons
- **Accuracy charts** (PNG): Precision difference visualizations (when using --output-diff)
- **Detailed reports** (Markdown): Comprehensive analysis with recommendations
- **Raw data** (JSON): Structured results for further analysis

## Testing

```bash
# Test framework functionality
python tests/test_extension.py

# Test framework with all operators
python run_comparator.py --operator matmul --test-cases small_square
python run_comparator.py --operator vector_add --test-cases small_tensor
python run_comparator.py --operator relu --test-cases small_tensor
```

### 5. Test Network Operators

```bash
# Test all network operators
python test_network_operators.py
```

## Profiling

```bash
# Nsight Systems profiling
./scripts/run_profiling.sh

# Nsight Compute profiling
./scripts/run_ncu_profiling.sh

# PyTorch-based profiling (when Nsight tools require special permissions)
python tools/profiling_pytorch.py

# Direct profiling tools usage
python tools/ncu_profiler.py --operator matmul --test-case small_square
python tools/profile_nsight.py --operator matmul --test-case small_square
```

## Requirements

### System Dependencies
- **Operating System**: Linux (Ubuntu, CentOS, RHEL, Alibaba Cloud Linux, etc.)
- **Compiler**: GCC with C++17 support
- **Build Tools**: CMake 3.10+
- **Python**: Python 3.8+ with development headers

### Python Dependencies
- **PyTorch** with CUDA support (required)
- **NumPy** (required)
- **Matplotlib** (optional, for visualization)
- **Seaborn** (optional, for enhanced visualization)

### Network Testing Dependencies (Optional)
- **iperf3**: For TCP bandwidth testing
- **RDMA tools**: InfiniBand verbs and perftest for RDMA testing
  - `infiniband-diags`
  - `libibverbs-dev` (Ubuntu) / `libibverbs-devel` (RHEL/CentOS)
  - `librdmacm-dev` (Ubuntu) / `librdmacm-devel` (RHEL/CentOS)
  - `perftest`
- **System utilities**: PCI utilities, NUMA control
  - `pciutils`
  - `numactl`
- **Network utilities**: netcat for network testing
  - `netcat-openbsd` (Ubuntu) / `nmap-ncat` (RHEL/CentOS)

### Hardware Requirements
- **NVIDIA GPU** with CUDA capability (for CUDA extensions)
- **InfiniBand/RoCE network** (for RDMA testing, optional)

### Automatic Installation
Use `./build.sh install-deps` to automatically detect and install missing dependencies on supported systems.

## Performance

The framework has been tested with various operator implementations and provides:
- Comprehensive performance comparisons
- Detailed analysis reports
- Visual performance charts
- Recommendations for optimal implementations

## Contributing

To add a new operator:

1. Create a new operator class in `src/operators/` inheriting from `BaseOperator`
2. Implement the required methods
3. Add test cases
4. Register the operator in the framework
5. Test with the universal comparator tool

See [FRAMEWORK_GUIDE.md](docs/FRAMEWORK_GUIDE.md) for detailed instructions.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Supported Operators

### Compute Operators
- **MatMul**: Matrix multiplication with CPU, PyTorch, custom CUDA implementations
- **VectorAdd**: Element-wise vector addition with CPU, PyTorch, custom CUDA implementations
- **ReLU**: Rectified Linear Unit activation with CPU, PyTorch, custom CUDA implementations
- **RMSNorm**: Root Mean Square Normalization with fused/unfused implementations

### Network Performance Operators
- **TCP Bandwidth**: TCP network bandwidth testing with multiple implementations
  - Client-server testing
  - iperf3 integration
  - netcat testing
- **RDMA Bandwidth**: RDMA network bandwidth testing
  - RDMA perftest integration
  - Ping-pong latency testing
  - Multi-QP testing
- **PCIe Bandwidth**: PCIe/GPU memory bandwidth testing
  - GPU memory copy operations
  - CUDA bandwidth testing
  - GPU peer-to-peer testing
  - NVLink testing
- **Network Stress**: Comprehensive network stress testing
  - Combined TCP/RDMA/PCIe testing
  - Concurrent connection testing
  - System-wide network benchmarking

## Tools Overview

For detailed information about the tools available in the `tools/` directory, refer to the [Tools README](tools/README.md).

## Troubleshooting

### Dependency Issues

If you encounter dependency-related errors:

1. **Check dependencies first**:
   ```bash
   ./build.sh check-deps
   ```

2. **Install missing dependencies automatically**:
   ```bash
   ./build.sh install-deps
   ```

3. **Manual dependency installation** (if automatic installation fails):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential cmake python3-dev iperf3 infiniband-diags libibverbs-dev librdmacm-dev perftest pciutils numactl netcat-openbsd
   
   # CentOS/RHEL/Alibaba Cloud Linux
   sudo yum install gcc-c++ cmake python3-devel iperf3 infiniband-diags libibverbs-devel librdmacm-devel perftest pciutils numactl nmap-ncat
   # Or for newer versions:
   sudo dnf install gcc-c++ cmake python3-devel iperf3 infiniband-diags libibverbs-devel librdmacm-devel perftest pciutils numactl nmap-ncat
   ```

### Network Testing Issues

- **RDMA tools not found**: Install InfiniBand utilities or skip RDMA tests
- **No RDMA devices**: Network tests will automatically skip RDMA-specific tests
- **Permission issues**: Some network tests may require sudo privileges
- **Firewall blocking**: Ensure network ports are accessible for bandwidth testing

### Build Issues

- **CUDA extension build fails**: Use `./build.sh --skip-cuda` for framework-only mode
- **Missing Python headers**: Install python3-dev (Ubuntu) or python3-devel (RHEL/CentOS)
- **Compiler not found**: Install build-essential (Ubuntu) or gcc-c++ (RHEL/CentOS)
- **Out of disk space**: Clean up space and retry, or use `./build.sh clean-all` to free up result files

### Cross-Server Deployment

When deploying to different servers:

1. **Copy the entire project directory** to the target server
2. **Run dependency check** on the new server: `./build.sh check-deps`
3. **Install missing dependencies**: `./build.sh install-deps`
4. **Build the framework**: `./build.sh`

The framework is designed to work across different Linux distributions and server configurations.