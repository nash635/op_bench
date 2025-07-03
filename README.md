# Universal Operator Benchmarking Framework

A flexible, extensible framework for implementing and comparing different operator implementations across various backends (PyTorch, CUDA, CuPy, etc.).

## Project Overview

This is a comprehensive operator benchmarking framework that provides high-performance implementations and detailed performance analysis across multiple backends including PyTorch, CuPy, and CUDA. The framework is designed for maximum compatibility and ease of use, with optional CUDA extension support for advanced users.

## 🏗️ Project Structure

```
universal_operator_framework/
├── 🔧 Core Framework (src/)
│   ├── framework/
│   │   ├── __init__.py
│   │   └── operator_framework.py      # Base framework classes and interfaces
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── matmul_operator.py         # Matrix multiplication implementations
│   │   ├── vector_add_operator.py     # Vector addition implementations
│   │   └── relu_operator.py           # ReLU activation implementations
│   └── cuda/
│       ├── __init__.py
│       ├── matmul_kernels.cu          # CUDA kernel implementations
│       ├── matmul_kernels.h           # CUDA kernel headers
│       └── matmul_cuda_ext.cpp        # PyTorch C++ bindings
├── 🛠️ Tools
│   └── operator_comparator_tool.py    # Universal CLI tool for benchmarking
├── 🧪 Tests
│   ├── test_extension.py              # Extension correctness tests
│   ├── ncu_profiler.py                # Nsight Compute profiling
│   ├── profile_nsight.py              # Nsight Systems profiling
│   └── profiling_alternative.py       # Alternative profiling methods
├── 📜 Scripts
│   ├── build.sh                       # Build script
│   ├── run_profiling.sh               # Nsight Systems profiling script
│   └── run_ncu_profiling.sh           # Nsight Compute profiling script
├── 📚 Documentation
│   ├── README.md                      # This document
│   ├── FRAMEWORK_GUIDE.md             # Detailed framework usage guide
│   ├── FRAMEWORK_ENHANCEMENT_SUMMARY.md # Framework enhancement summary
│   ├── CLEANUP_SUMMARY.md             # Project cleanup summary
│   └── DIRECTORY_RESTRUCTURE_SUMMARY.md # Directory restructure summary
├── 🚀 Convenience Scripts
│   ├── build.sh                       # Convenience build script
│   ├── run_comparator.py              # Main entry point for comparator
│   └── setup.py                       # Extension build configuration
└── 📦 Build Artifacts
    ├── build/                         # Build cache directory
    ├── __pycache__/                   # Python cache
    └── *.so                           # Compiled extension library
```

## 🚀 Quick Start

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
```

## 📊 Supported Operators

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

## 🔧 Framework Features

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

## 📈 Output Examples

The framework generates:
- **Performance charts** (PNG): Execution time and GFLOPS comparisons
- **Detailed reports** (Markdown): Comprehensive analysis with recommendations
- **Raw data** (JSON): Structured results for further analysis

## 🧪 Testing

```bash
# Test framework functionality
python tests/test_extension.py

# Test framework with all operators
python run_comparator.py --operator matmul --test-cases small_square
python run_comparator.py --operator vector_add --test-cases small_tensor
python run_comparator.py --operator relu --test-cases small_tensor
```

## 🔬 Profiling

```bash
# Nsight Systems profiling
./scripts/run_profiling.sh

# Nsight Compute profiling
./scripts/run_ncu_profiling.sh

# Alternative profiling methods
python tests/profiling_alternative.py
```

## 📚 Documentation

- **[FRAMEWORK_GUIDE.md](docs/FRAMEWORK_GUIDE.md)**: Detailed framework usage and extension guide
- **[FRAMEWORK_ENHANCEMENT_SUMMARY.md](docs/FRAMEWORK_ENHANCEMENT_SUMMARY.md)**: Summary of framework enhancements
- **[FINAL_CLEANUP_SUMMARY.md](docs/FINAL_CLEANUP_SUMMARY.md)**: Final project cleanup summary
- **[DIRECTORY_RESTRUCTURE_SUMMARY.md](docs/DIRECTORY_RESTRUCTURE_SUMMARY.md)**: Directory restructure summary

## 🛠️ Requirements

- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA capability
- Optional: CuPy, matplotlib, seaborn

## 📊 Performance

The framework has been tested with various operator implementations and provides:
- Comprehensive performance comparisons
- Detailed analysis reports
- Visual performance charts
- Recommendations for optimal implementations

## 🤝 Contributing

To add a new operator:

1. Create a new operator class in `src/operators/` inheriting from `BaseOperator`
2. Implement the required methods
3. Add test cases
4. Register the operator in the framework
5. Test with the universal comparator tool

See [FRAMEWORK_GUIDE.md](docs/FRAMEWORK_GUIDE.md) for detailed instructions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔄 Recent Changes

### Latest Cleanup (July 2025)
- **Removed redundant matmul_comparator.py**: Functionality fully integrated into universal tool
- **Cleaned temporary output files**: Removed test-generated comparison files
- **Simplified tools directory**: Now contains only the universal comparator tool
- **Maintained full functionality**: All MatMul features available through universal tool

### Directory Structure Reorganization
- **Organized source code**: Moved all source files to `src/` directory
- **Separated concerns**: Framework, operators, CUDA implementation, tools, tests, scripts, and docs are now in separate directories
- **Added convenience scripts**: `run_comparator.py` and `build.sh` in root for easy access
- **Updated import paths**: All imports now use the new structured paths
- **Maintained functionality**: All existing features work with the new structure

### Benefits of Current Structure
- **Single universal tool**: One tool handles all operators instead of separate tools per operator
- **Better organization**: Clear separation of different types of files
- **Easier maintenance**: Logical grouping makes it easier to find and modify code
- **Scalability**: New structure supports growth and additional features
- **Professional appearance**: Follows Python package best practices
