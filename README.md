# Universal Operator Benchmarking Framework

A flexible, extensible framework for implementing and comparing different operator implementations (perf/output...) across various backends (PyTorch, CUDA, CuPy, etc.).


## Quick Start

### 1. Build the Framework

```bash
# Simple build (attempts CUDA extension - recommended)
./build.sh

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
./build.sh clean-all               # Clean build artifacts and experimental output
./build.sh test-framework           # Test framework functionality
./build.sh check-deps               # Check dependencies only
```

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

## Supported Operators

### Matrix Multiplication (MatMul)
- **PyTorch**: torch.mm, torch.addmm, torch.bmm
- **CUDA Kernels**: Basic, shared memory, template-based
- **CuPy**: cupy.matmul

### Vector Addition
- **PyTorch**: torch.add
- **CUDA Kernels**: Basic element-wise addition
- **CuPy**: cupy.add
- **NumPy**: numpy.add

### ReLU Activation
- **PyTorch**: torch.relu, torch.nn.functional.relu
- **CUDA Kernels**: Custom element-wise ReLU
- **CuPy**: cupy.maximum with zero

### RMSNorm (Root Mean Square Normalization)
- **Fused implementations**: Semi-fused, custom CUDA kernels
- **Unfused implementations**: Step-by-step PyTorch operations
- **Comparison targets**: LayerNorm, explicit intermediate storage
- **Mathematical equivalence**: All implementations produce identical results

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

## Profiling

```bash
# Nsight Systems profiling
./scripts/run_profiling.sh

# Nsight Compute profiling
./scripts/run_ncu_profiling.sh

# Alternative profiling methods
python tests/profiling_alternative.py
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability
- Optional: CuPy, matplotlib, seaborn

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