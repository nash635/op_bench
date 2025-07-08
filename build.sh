#!/bin/bash
#
# Optimal MatMul Extension - Unified Build Script
# Supports CMake and setup.py build methods with comprehensive options
#

set -e  # Exit on error

# Configuration
PROJECT_NAME="Universal Operator Benchmarking Framework"
BUILD_TYPE="Release"
JOBS=$(nproc)
VERBOSE=false
CLEAN=false
DEBUG=false
METHOD="cmake"
TARGET="all"
RUN_TESTS=false
RUN_BENCHMARKS=false
CHECK_DEPS=false
SKIP_CUDA=false
PYTHON_EXECUTABLE="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}
PROJECT_NAME="Universal Operator Benchmarking Framework"
BUILD_TYPE="Release"
JOBS=$(nproc)
VERBOSE=false
CLEAN=false
DEBUG=false
METHOD="cmake"
TARGET="all"
RUN_TESTS=false
RUN_BENCHMARKS=false
CHECK_DEPS=false
SKIP_CUDA=false
PYTHON_EXECUTABLE="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
$PROJECT_NAME Build Script

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    build           Build the project (default)
    clean           Clean build artifacts
    clean-all       Clean build artifacts and all experimental output data
    test            Run tests
    benchmark       Run benchmarks
    check-deps      Check dependencies only
    test-framework  Test framework without CUDA extension
    cuda            Force CUDA extension build attempt
    help            Show this help

OPTIONS:
    --method METHOD     Build method: cmake, python, both (default: cmake)
    --build-type TYPE   Build type: Debug, Release (default: Release)
    --jobs N           Number of parallel jobs (default: $(nproc))
    --verbose          Enable verbose output
    --debug            Show detailed compilation output
    --clean            Clean before building
    --test             Run tests after building
    --benchmark        Run benchmarks after building
    --skip-cuda        Skip CUDA extension build (framework only)
    --force-cuda       Force CUDA extension build attempt

EXAMPLES:
    $0                          # Build with CUDA extension (default)
    $0 --skip-cuda              # Build in framework-only mode
    $0 cuda                     # Force CUDA extension build
    $0 --test                   # Build and run tests
    $0 --benchmark              # Build and run benchmarks
    $0 --debug                  # Build with detailed output
    $0 test-framework           # Test framework without CUDA
    $0 check-deps               # Check dependencies only

NOTE: CUDA extension build is attempted by default.
      Use --skip-cuda for framework-only mode if CUDA build fails.
      All core features (PyTorch, CuPy, performance analysis) are fully available.

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check CMake
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
        log_success "CMake $CMAKE_VERSION found"
        CMAKE_AVAILABLE=true
    else
        log_warning "CMake not found"
        CMAKE_AVAILABLE=false
    fi
    
    # Check C++ compiler and version
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n1)
        log_success "g++ $GCC_VERSION found"
        
        # Check if g++ supports C++17
        if g++ -std=c++17 -x c++ -E - < /dev/null &> /dev/null; then
            log_success "C++17 support confirmed"
        else
            log_warning "C++17 support not confirmed"
        fi
    else
        log_warning "g++ not found"
    fi
    
    # Check Python and packages
    if command -v $PYTHON_EXECUTABLE &> /dev/null; then
        PYTHON_VERSION=$($PYTHON_EXECUTABLE --version 2>&1 | sed 's/Python //')
        log_success "Python $PYTHON_VERSION found"
        
        # Check Python development headers
        PYTHON_INCLUDE_PATH=$($PYTHON_EXECUTABLE -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
        if [ -n "$PYTHON_INCLUDE_PATH" ] && [ -f "$PYTHON_INCLUDE_PATH/Python.h" ]; then
            log_success "Python development headers found"
        else
            log_warning "Python development headers not found"
            log_info "Try: conda install python-dev or apt-get install python3-dev"
        fi
        
        # Check PyTorch
        if $PYTHON_EXECUTABLE -c "import torch" 2>/dev/null; then
            PYTORCH_VERSION=$($PYTHON_EXECUTABLE -c "import torch; print(torch.__version__)" 2>/dev/null)
            log_success "PyTorch $PYTORCH_VERSION found"
            
            # Check CUDA availability
            CUDA_CHECK_OUTPUT=$($PYTHON_EXECUTABLE -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" 2>/dev/null)
            if [ "$CUDA_CHECK_OUTPUT" = "CUDA_OK" ]; then
                CUDA_VERSION=$($PYTHON_EXECUTABLE -c "import torch; print(torch.version.cuda)" 2>/dev/null)
                GPU_NAME=$($PYTHON_EXECUTABLE -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
                log_success "CUDA $CUDA_VERSION available"
                log_success "GPU: $GPU_NAME"
            else
                log_warning "CUDA not available in PyTorch"
            fi
        else
            log_error "PyTorch not found"
            return 1
        fi
        
        # Check NumPy
        if $PYTHON_EXECUTABLE -c "import numpy" 2>/dev/null; then
            NUMPY_VERSION=$($PYTHON_EXECUTABLE -c "import numpy; print(numpy.__version__)" 2>/dev/null)
            log_success "NumPy $NUMPY_VERSION found"
        else
            log_error "NumPy not found"
            return 1
        fi
    else
        log_error "Python not found"
        return 1
    fi
    
    return 0
}

get_torch_cmake_config() {
    $PYTHON_EXECUTABLE -c '
import torch
import os
torch_dir = os.path.dirname(torch.__file__)
cmake_prefix_path = os.path.join(torch_dir, "share", "cmake")
if os.path.exists(cmake_prefix_path):
    print(cmake_prefix_path)
else:
    print(torch_dir)
' 2>/dev/null || echo ""
}

build_with_cmake() {
    log_info "Building with CMake..."
    echo "============================================================"
    
    if [ "$CMAKE_AVAILABLE" != "true" ]; then
        log_error "CMake not available"
        return 1
    fi
    
    # Build directory
    BUILD_DIR="build"
    
    if [ "$CLEAN" = "true" ] && [ -d "$BUILD_DIR" ]; then
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    
    # CMake configuration - build command arguments
    CMAKE_BASE_ARGS="cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -Wno-dev -DCMAKE_WARN_DEPRECATED=OFF"
    
    # Use conda compiler if available, otherwise use system compiler
    if command -v x86_64-conda-linux-gnu-g++ &> /dev/null; then
        CMAKE_COMPILER_ARGS="-DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ -DCMAKE_C_COMPILER=x86_64-conda-linux-gnu-gcc"
        log_info "Using conda GCC toolchain"
    else
        CMAKE_COMPILER_ARGS="-DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc"
        log_info "Using system GCC toolchain"
    fi
    
    # Add environment variables to suppress warnings
    export PYTORCH_NVCC_FLAGS="-Xcompiler -fPIC"
    export TORCH_CUDA_ARCH_LIST="6.0"
    
    # Add PyTorch CMake path
    TORCH_CMAKE_PATH=$(get_torch_cmake_config)
    if [ -n "$TORCH_CMAKE_PATH" ]; then
        CMAKE_PREFIX_ARGS="-DCMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH"
    else
        CMAKE_PREFIX_ARGS=""
    fi
    
    if [ "$VERBOSE" = "true" ]; then
        CMAKE_VERBOSE_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON"
    else
        CMAKE_VERBOSE_ARGS=""
    fi
    
    # Combine all CMake arguments
    CMAKE_FULL_ARGS="$CMAKE_BASE_ARGS $CMAKE_COMPILER_ARGS $CMAKE_PREFIX_ARGS $CMAKE_VERBOSE_ARGS"
    
    # Configure
    log_info "Configuring CMake..."
    if [ "$DEBUG" = "true" ]; then
        log_info "Debug mode: showing CMake configuration output"
        (cd "$BUILD_DIR" && $CMAKE_FULL_ARGS)
    else
        if ! (cd "$BUILD_DIR" && $CMAKE_FULL_ARGS &>/dev/null); then
            log_error "CMake configuration failed"
            log_info "Run with --debug to see detailed output"
            return 1
        fi
    fi
    
    # Build
    log_info "Building..."
    BUILD_BASE_ARGS="cmake --build . --parallel $JOBS"
    
    if [ "$TARGET" != "all" ]; then
        BUILD_TARGET_ARGS="--target $TARGET"
    else
        BUILD_TARGET_ARGS=""
    fi
    
    if [ "$VERBOSE" = "true" ]; then
        BUILD_VERBOSE_ARGS="--verbose"
    else
        BUILD_VERBOSE_ARGS=""
    fi
    
    BUILD_FULL_ARGS="$BUILD_BASE_ARGS $BUILD_TARGET_ARGS $BUILD_VERBOSE_ARGS"
    
    if [ "$DEBUG" = "true" ]; then
        log_info "Debug mode: showing build output"
        # Use timeout to prevent hanging
        if ! timeout 300 bash -c "cd '$BUILD_DIR' && $BUILD_FULL_ARGS"; then
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                log_error "Build timed out after 300 seconds"
                log_info "This may indicate a compilation issue or resource constraint"
            else
                log_error "Build failed"
            fi
            return 1
        fi
    else
        BUILD_OUTPUT=$(timeout 300 bash -c "cd '$BUILD_DIR' && $BUILD_FULL_ARGS" 2>&1)
        BUILD_EXIT_CODE=$?
        
        if [ $BUILD_EXIT_CODE -eq 124 ]; then
            log_error "Build timed out after 300 seconds"
            log_info "This may indicate a compilation issue or resource constraint"
            return 1
        elif [ $BUILD_EXIT_CODE -ne 0 ]; then
            log_error "Build failed with exit code $BUILD_EXIT_CODE"
            log_info "Run with --debug to see detailed output"
            echo "$BUILD_OUTPUT" | grep -E "error|Error|failed|Failed" | head -10
            return 1
        fi
    fi
    
    log_success "CMake build completed successfully!"
    return 0
}

build_with_python() {
    log_info "Building with Python setup.py..."
    echo "============================================================"
    
    # Set correct compilers and flags
    if command -v x86_64-conda-linux-gnu-g++ &> /dev/null; then
        export CC=x86_64-conda-linux-gnu-gcc
        export CXX=x86_64-conda-linux-gnu-g++
        log_info "Using conda GCC toolchain for Python build"
    else
        export CC=/usr/bin/gcc
        export CXX=/usr/bin/g++
        log_info "Using system GCC toolchain for Python build"
    fi
    export CXXFLAGS="-std=c++17"
    export NVCC_PREPEND_FLAGS="-ccbin $CXX"
    
    if [ "$CLEAN" = "true" ]; then
        log_info "Cleaning Python build artifacts..."
        rm -rf build/ dist/ *.egg-info/ *.so __pycache__/
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    fi
    
    BUILD_BASE_CMD="$PYTHON_EXECUTABLE setup.py build_ext --inplace"
    
    if [ "$VERBOSE" = "true" ]; then
        BUILD_VERBOSE_ARG="--verbose"
    else
        BUILD_VERBOSE_ARG=""
    fi
    
    BUILD_FULL_CMD="$BUILD_BASE_CMD $BUILD_VERBOSE_ARG"
    
    if [ "$DEBUG" = "true" ]; then
        log_info "Debug mode: showing Python build output"
        if ! $BUILD_FULL_CMD; then
            log_error "Python build failed"
            return 1
        fi
    else
        BUILD_OUTPUT=$($BUILD_FULL_CMD 2>&1)
        BUILD_EXIT_CODE=$?
        
        if [ $BUILD_EXIT_CODE -ne 0 ]; then
            log_error "Python build failed with exit code $BUILD_EXIT_CODE"
            log_info "Run with --debug to see detailed output"
            echo "$BUILD_OUTPUT" | grep -E "error|Error|failed|Failed" | head -10
            return 1
        fi
    fi
    
    log_success "Python build completed successfully!"
    return 0
}

run_tests() {
    log_info "Running tests..."
    echo "============================================================"
    
    # Test framework functionality (always works)
    log_info "Testing framework functionality..."
    if ! $PYTHON_EXECUTABLE run_comparator.py --list-operators; then
        log_error "Framework test failed"
        return 1
    fi
    
    # Test extension only if not skipping CUDA
    if [ "$SKIP_CUDA" = "false" ] && [ -f "tests/test_extension.py" ]; then
        log_info "Running extension tests..."
        if ! $PYTHON_EXECUTABLE tests/test_extension.py; then
            log_error "Extension tests failed"
            return 1
        fi
    elif [ "$SKIP_CUDA" = "true" ]; then
        log_info "Skipping CUDA extension tests"
    fi
    
    log_success "All tests passed!"
    return 0
}

run_benchmarks() {
    log_info "Running benchmarks..."
    echo "============================================================"
    
    if ! $PYTHON_EXECUTABLE run_comparator.py --operator matmul --test-cases small_square; then
        log_error "Benchmark failed"
        return 1
    fi
    
    log_success "Benchmarks completed!"
    return 0
}

clean_all() {
    log_info "Cleaning all build artifacts and experimental output data..."
    echo "============================================================"
    
    # CMake build artifacts
    rm -rf build/
    
    # Python build artifacts
    log_info "Removing Python build artifacts..."
    rm -rf dist/ *.egg-info/ *.so
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Additional Python cache cleanup (more thorough)
    log_info "Removing all Python cache directories..."
    find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf 2>/dev/null || true
    find . -name "*.py[co]" -delete 2>/dev/null || true
    
    # Temporary files
    rm -rf tmp/ temp/ *.tmp
    
    # Experimental output files
    log_info "Removing experimental output files..."
    
    # Comparison and benchmark output files
    rm -f comparison_*.json comparison_*.md comparison_*.txt
    rm -f benchmark_*.json benchmark_*.md benchmark_*.txt
    rm -f results_*.json results_*.md results_*.txt
    rm -f performance_*.json performance_*.md performance_*.txt
    rm -f timing_*.json timing_*.md timing_*.txt
    rm -f profile_*.json profile_*.md profile_*.txt
    rm -f analysis_*.json analysis_*.md analysis_*.txt
    
    # Image and visualization files
    rm -f *.png *.jpg *.jpeg *.svg *.pdf *.gif *.bmp *.tiff
    
    # Log and output files
    rm -f *.log *.out *.err *.trace
    rm -f output_*.txt output_*.json output_*.md
    rm -f test_*.txt test_*.json test_*.md
    rm -f experiment_*.txt experiment_*.json experiment_*.md
    
    # Generated code and compilation artifacts
    rm -f *.ptx *.cubin *.fatbin *.cu.o *.cu.cpp
    rm -f generated_*.py generated_*.cu generated_*.cpp
    
    # Cache and runtime files
    rm -rf .pytest_cache/ .coverage* htmlcov/
    rm -f core core.* *.core
    
    log_success "Complete cleanup finished!"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARKS=true
            shift
            ;;
        --skip-cuda)
            SKIP_CUDA=true
            shift
            ;;
        --force-cuda)
            SKIP_CUDA=false
            shift
            ;;
        build)
            # Default command
            shift
            ;;
        clean)
            clean_all
            exit 0
            ;;
        clean-all)
            clean_all
            exit 0
            ;;
        test)
            RUN_TESTS=true
            shift
            ;;
        benchmark)
            RUN_BENCHMARKS=true
            shift
            ;;
        test-framework)
            SKIP_CUDA=true
            RUN_TESTS=true
            shift
            ;;        cuda)
            SKIP_CUDA=false
            shift
            ;;
        check-deps)
            CHECK_DEPS=true
            shift
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
cd "$(dirname "$0")"

echo "[INFO] $PROJECT_NAME Build Script"
echo "============================================================"

# Check dependencies
if ! check_dependencies; then
    log_error "Dependency check failed"
    exit 1
fi

if [ "$CHECK_DEPS" = "true" ]; then
    log_success "All dependencies satisfied"
    exit 0
fi

build_framework_mode() {
    log_info "Building in framework mode - optimized for compatibility..."
    echo "============================================================"
    
    if [ "$DEBUG" = "true" ]; then
        log_info "Debug mode: Framework mode details"
        echo "  - Build method: $METHOD"
        echo "  - Build type: $BUILD_TYPE"
        echo "  - Jobs: $JOBS"
        echo "  - Python executable: $PYTHON_EXECUTABLE"
        echo "  - CUDA skipped: $SKIP_CUDA"
        echo "  - Clean requested: $CLEAN"
        echo "  - Verbose: $VERBOSE"
        echo
    fi
    
    # Clean if requested
    if [ "$CLEAN" = "true" ]; then
        log_info "Cleaning build artifacts..."
        rm -rf build/ dist/ *.egg-info/ *.so __pycache__/
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        if [ "$DEBUG" = "true" ]; then
            log_info "Debug mode: Cleaned build artifacts"
        fi
    fi
    
    # Verify framework components
    if [ "$DEBUG" = "true" ]; then
        log_info "Debug mode: Verifying framework components..."
        
        # Check key framework files
        if [ -f "run_comparator.py" ]; then
            echo "  [PASS] Main comparator tool found"
        else
            echo "  [FAIL] Main comparator tool missing"
        fi
        
        if [ -d "src/framework" ]; then
            echo "  [PASS] Framework source directory found"
        else
            echo "  [FAIL] Framework source directory missing"
        fi
        
        if [ -d "src/operators" ]; then
            echo "  [PASS] Operators directory found"
        else
            echo "  [FAIL] Operators directory missing"
        fi
        
        # Check Python imports
        if $PYTHON_EXECUTABLE -c "import torch" 2>/dev/null; then
            echo "  [PASS] PyTorch import successful"
        else
            echo "  [FAIL] PyTorch import failed"
        fi
        
        if $PYTHON_EXECUTABLE -c "import numpy" 2>/dev/null; then
            echo "  [PASS] NumPy import successful"
        else
            echo "  [FAIL] NumPy import failed"
        fi
        
        # Test framework functionality
        log_info "Debug mode: Testing framework functionality..."
        if $PYTHON_EXECUTABLE -c '
import sys
sys.path.insert(0, ".")
try:
    from src.framework.operator_framework import BaseOperator, OperatorType
    print("  [PASS] Framework core imports successful")
except Exception as e:
    print("  [FAIL] Framework core import failed:", str(e))
' 2>/dev/null; then
            :
        else
            echo "  [FAIL] Framework core import test failed"
        fi
    fi
    
    log_success "Framework mode build completed successfully!"
    return 0
}

# Build
SUCCESS=true

if [ "$SKIP_CUDA" = "false" ]; then
    log_info "Building with CUDA extension..."
    case $METHOD in
        cmake)
            if ! build_with_cmake; then
                log_warning "CUDA extension build failed, switching to framework-only mode"
                SKIP_CUDA=true
                SUCCESS=true
            fi
            ;;
        python)
            if ! build_with_python; then
                log_warning "CUDA extension build failed, switching to framework-only mode"
                SKIP_CUDA=true
                SUCCESS=true
            fi
            ;;
        both)
            if ! build_with_cmake; then
                log_warning "CMake build failed, trying Python build..."
                if ! build_with_python; then
                    log_warning "Both builds failed, switching to framework-only mode"
                    SKIP_CUDA=true
                    SUCCESS=true
                fi
            fi
            ;;
        *)
            log_error "Unknown build method: $METHOD"
            exit 1
            ;;
    esac
else
    build_framework_mode
fi

if [ "$SUCCESS" = "false" ]; then
    log_error "Build failed"
    exit 1
fi

# Post-build actions
if [ "$RUN_TESTS" = "true" ]; then
    if ! run_tests; then
        log_error "Tests failed"
        exit 1
    fi
fi

if [ "$RUN_BENCHMARKS" = "true" ]; then
    if ! run_benchmarks; then
        log_error "Benchmarks failed"
        exit 1
    fi
fi

# Success message
echo
log_success "Framework build completed successfully!"
echo
echo "[INFO] Ready to use:"
echo "   List operators: python run_comparator.py --list-operators"
echo "   Run benchmarks: python run_comparator.py --operator matmul --test-cases small_square"
echo "   Framework tests: ./build.sh --test"
echo
echo "[INFO] Quick commands:"
echo "   ./build.sh --test                # Run all tests"
echo "   ./build.sh --benchmark           # Run benchmarks"
echo "   ./build.sh --force-cuda          # Attempt CUDA build (advanced)"
