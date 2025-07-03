cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(optimal_matmul_extension LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Suppress developer warnings for external packages
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Find Python and get include directory  
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")

# Force C++17 for CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
endif()

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(src/cuda)

# Source files
set(SOURCES
    src/cuda/matmul_cuda_ext.cpp
    src/cuda/matmul_kernels.cu
)

# Create the extension library
add_library(matmul_cuda_ext SHARED ${SOURCES})

# Set properties for the library
set_target_properties(matmul_cuda_ext PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(matmul_cuda_ext ${TORCH_LIBRARIES})
target_link_libraries(matmul_cuda_ext CUDA::cudart CUDA::cublas)

# Set CUDA architectures (adjust based on your GPU)
set_property(TARGET matmul_cuda_ext PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)

# Compiler-specific options
target_compile_definitions(matmul_cuda_ext PRIVATE ${TORCH_CXX_FLAGS})

# Custom target for Python integration
add_custom_target(python_build
    COMMAND ${CMAKE_COMMAND} -E echo "Building Python extension..."
    COMMAND python setup.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Python extension using setup.py"
)

# Custom target for testing
add_custom_target(test
    COMMAND python tests/test_extension.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS matmul_cuda_ext
    COMMENT "Running extension tests"
)

# Custom target for benchmarking
add_custom_target(benchmark
    COMMAND python run_comparator.py --operator matmul --test-cases small_square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS matmul_cuda_ext
    COMMENT "Running performance benchmarks"
)

# Custom target for cleaning
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory build
    COMMAND ${CMAKE_COMMAND} -E remove *.so
    COMMAND ${CMAKE_COMMAND} -E remove_directory __pycache__
    COMMAND find . -name "*.pyc" -delete 2>/dev/null || true
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning all build artifacts"
)

# Install target
install(TARGETS matmul_cuda_ext
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "Optimal MatMul Extension Configuration")
message(STATUS "==========================================")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
message(STATUS "TORCH_VERSION: ${Torch_VERSION}")
message(STATUS "CUDA_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUTLASS_FOUND: ${CUTLASS_FOUND}")
if(CUTLASS_FOUND)
    message(STATUS "CUTLASS_PATH: ${CUTLASS_PATH}")
endif()
message(STATUS "==========================================")
