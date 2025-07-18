# FP8 Linear Operator Documentation

## Overview

The FP8 Linear Operator provides high-performance linear layer implementations optimized for modern GPUs, particularly H100 and B200 class accelerators. It supports multiple backends and includes comprehensive test cases designed to maximize computational throughput.

## Features

- Multiple FP8 backend implementations (PyTorch BF16 baseline, optional Transformer Engine, CUTLASS, cuBLAS)
- High-performance test cases optimized for H100/B200 GPUs
- Real-world LLM workload scenarios (LLaMA3-405B, GPT-4 scale)
- Memory bandwidth optimization tests
- Automatic GPU capability detection and adaptive benchmarking
- Automatic chart generation with --plot flag
- Pre-configured test scenarios for different GPU classes
- Analysis tips and guidance

## Test Cases

### Quick Validation
- `quick_validation`: 256×512 → 1024 (0.27 GFLOPs) - Basic functionality test
- `medium_baseline`: 1024×2048 → 4096 (16.78 GFLOPs) - Medium-scale baseline

### H100 Optimized Test Cases
- `h100_target_small`: 8192×8192 → 8192 (1.07 TFLOPs) - H100 small target
- `h100_target_medium`: 16384×12288 → 16384 (6.44 TFLOPs) - H100 medium target  
- `h100_target_large`: 32768×16384 → 32768 (35.18 TFLOPs) - H100 large target
- `h100_peak_stress`: 65536×24576 → 49152 (157.29 TFLOPs) - H100 peak stress test

### B200 Optimized Test Cases
- `b200_target_medium`: 65536×32768 → 65536 (274.88 TFLOPs) - B200 medium target
- `b200_target_large`: 131072×49152 → 98304 (1.23 PFLOPs) - B200 large target
- `b200_peak_stress`: 262144×65536 → 131072 (4.40 PFLOPs) - B200 ultimate stress test

### Real-World LLM Workloads
- `llama3_405b_ffn_up`: 32768×16384 → 53248 (35.18 TFLOPs) - LLaMA3-405B FFN up projection
- `llama3_405b_ffn_down`: 32768×53248 → 16384 (57.27 TFLOPs) - LLaMA3-405B FFN down projection
- `gpt4_scale_attention`: 131072×32768 → 98304 (824.63 TFLOPs) - GPT-4 scale attention

### Memory Bandwidth Tests
- `memory_bandwidth_square`: 32768×32768 → 32768 (70.37 TFLOPs) - Square matrix optimization
- `memory_bandwidth_wide`: 16384×131072 → 16384 (70.37 TFLOPs) - Wide matrix test
- `memory_bandwidth_tall`: 131072×16384 → 16384 (70.37 TFLOPs) - Tall matrix test

## Performance Targets

### H100 SXM (1000 TFLOPS FP8 theoretical peak)
- Target efficiency: 60-80% of theoretical peak
- Optimal test cases: `h100_target_*` series
- Peak stress test: `h100_peak_stress` (157+ TFLOPs computational workload)

### B200 (2500 TFLOPS FP8 theoretical peak)  
- Target efficiency: 70-85% of theoretical peak
- Optimal test cases: `b200_target_*` series
- Peak stress test: `b200_peak_stress` (4+ PFLOPs computational workload)

## Usage

### Command Line Interface

```bash
# Auto-detect GPU and run optimal tests
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive --plot

# Use predefined H100 configuration
python tools/operator_comparator_tool.py --config h100_performance_suite --plot

# Profile GPU capabilities only
python tools/operator_comparator_tool.py --adaptive --profile-only

# Run specific test cases with plotting
python tools/operator_comparator_tool.py --operator fp8_linear --test-cases quick_validation medium_baseline --plot

# Run with custom parameters
python tools/operator_comparator_tool.py --operator fp8_linear --warmup 5 --runs 10 --plot
```

**Note**: All results are saved to the `test_results` directory by default. Use `--output-dir custom_path` to specify a different location.

#### Available Backends

The FP8 Linear operator supports the following backends with automatic data type optimization:

1. **pytorch_bf16**: PyTorch BF16 baseline (always available)
   - Input/Compute: bfloat16
   - Use case: Reference implementation and compatibility baseline
   
2. **transformer_engine**: NVIDIA Transformer Engine FP8 (if installed)
   - Input: bfloat16, Compute: FP8 (automatic scaling)
   - Use case: Production FP8 inference with automatic scaling
   
3. **cutlass**: Custom CUTLASS FP8 implementation (if available)
   - Input/Compute: float8_e4m3fn (automatic conversion)
   - Use case: Custom high-performance FP8 GEMM kernels
   
4. **cublas**: Custom cuBLAS FP8 implementation (if available)
   - Input/Compute: float8_e4m3fn (automatic conversion)
   - Use case: cuBLAS-optimized FP8 linear operations

The system automatically detects which backends are available in your environment and performs appropriate data type conversions for optimal performance. No simulated or mock backends are included.

#### Adaptive Mode

Use `--adaptive` for automatic GPU detection and optimal test case selection:

```bash
# Automatically select optimal test cases based on GPU
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive --plot

# Include stress tests for high-memory GPUs
python tools/operator_comparator_tool.py --operator fp8_linear --adaptive --stress-tests --plot
```

#### Predefined Configurations

Choose from predefined configurations optimized for specific GPU classes:

```bash
# H100 configurations
python tools/operator_comparator_tool.py --config h100_quick_validation --plot
python tools/operator_comparator_tool.py --config h100_performance_suite --plot
python tools/operator_comparator_tool.py --config h100_peak_stress --plot

# B200 configurations  
python tools/operator_comparator_tool.py --config b200_performance_suite --plot
python tools/operator_comparator_tool.py --config b200_peak_stress --plot
```

#### Chart Generation

When using `--plot`, the tool generates:
- **Performance GFLOPS Chart**: Compares computational throughput across available backends
- **Execution Time Chart**: Compares execution times across different problem sizes
- **Detailed Reports**: Markdown and JSON formats with comprehensive results

### Available Test Cases

The operator includes optimized test cases for different performance scenarios:

1. **quick_validation**: 256×512 → 1024 (0.27 GFLOPs) - Basic functionality test
2. **medium_baseline**: 1024×2048 → 4096 (16.78 GFLOPs) - Medium-scale baseline
3. **h100_target_small**: 8192×8192 → 8192 (1.07 TFLOPs) - H100 small target
4. **h100_target_medium**: 16384×12288 → 16384 (6.44 TFLOPs) - H100 medium target
5. **h100_target_large**: 32768×16384 → 32768 (35.18 TFLOPs) - H100 large target
6. **h100_peak_stress**: 65536×24576 → 49152 (157.29 TFLOPs) - H100 peak stress
7. **b200_target_medium**: 65536×32768 → 65536 (274.88 TFLOPs) - B200 medium target
8. **b200_target_large**: 131072×49152 → 98304 (1.23 PFLOPs) - B200 large target
9. **b200_peak_stress**: 262144×65536 → 131072 (4.40 PFLOPs) - B200 ultimate stress
10. **llama3_405b_ffn_up/down**: LLaMA3-405B scale FFN projections
11. **gpt4_scale_attention**: GPT-4 scale attention (824+ TFLOPs)
12. **memory_bandwidth_***: Memory bandwidth optimization tests

### Python API Example

```python
import sys
import os
sys.path.insert(0, 'src')

from operators.fp8_linear_operator import FP8LinearOperator
from tools.operator_comparator_tool import UniversalOperatorComparator

# Create and register the operator
operator = FP8LinearOperator()
comparator = UniversalOperatorComparator()
comparator.register_operator(operator)

# Run benchmarks with available backends only
results, outputs = comparator.run(
    operator_type="fp8_linear",
    implementations=None,  # Use all available backends
    test_cases=["quick_validation", "medium_baseline"],
    warmup_runs=5,
    test_runs=10
)

# Print results
for test_case, impl_results in results.items():
    print(f"\nTest Case: {test_case}")
    for result in impl_results:
        if result.available:
            print(f"  {result.impl_id}: {result.display_metric}")
```

## Implementation Details

### FLOPs Calculation Logic

#### Unit Terminology Clarification

- **FLOPs** (Floating-point Operations): The total number of floating-point operations required for a computation
- **FLOPS** (Floating-point Operations Per Second): The rate at which floating-point operations are performed

In our benchmarking, the values shown in test case descriptions represent **computational complexity** (FLOPs), not throughput rate (FLOPS).

#### Calculation Method

For a linear layer operation `Y = X @ W`, where:
- `X` has shape `(batch_size, in_features)`
- `W` has shape `(in_features, out_features)`
- `Y` has shape `(batch_size, out_features)`

The total FLOPs calculation is:
```
Total FLOPs = 2 × batch_size × in_features × out_features
```

**Explanation:**
- Each output element `Y[i,j]` requires `in_features` multiply-add operations
- Factor of 2 accounts for both multiplication and addition in each MAC operation
- Total output elements: `batch_size × out_features`
- Therefore: `2 × batch_size × in_features × out_features`

#### Example Calculation

For `llama3_405b_ffn_down` test case:
- Input shape: `(32768, 53248)` → `batch_size=32768, in_features=53248`
- Output features: `16384` → `out_features=16384`
- Calculation: `2 × 32768 × 53248 × 16384 = 57,174,604,644,352 FLOPs`
- In TFLOPs: `57.175 TFLOPs`

This represents the **computational workload** of the operation, not the execution speed.

### Backend Detection

The operator automatically detects available backends:

- **pytorch_bf16**: Always available (baseline implementation)
- **transformer_engine**: Available if `transformer_engine` package is installed
- **cutlass**: Available if custom `linear` module with CUTLASS support is importable
- **cublas**: Available if custom `linear` module with cuBLAS support is importable

Only real, functional backends are registered and used. The system does not include any simulated or mock implementations.

### Memory and Performance

- **Data Type Handling**: Automatic backend-specific data type selection and conversion
  - `pytorch_bf16`: Uses bfloat16 for both input and computation
  - `transformer_engine`: Uses bfloat16 input, FP8 computation internally
  - `cutlass`: Converts input to float8_e4m3fn for optimal FP8 performance
  - `cublas`: Converts input to float8_e4m3fn for optimal FP8 performance
- Input tensors are generated with appropriate ranges for each data type
- FLOPs calculation: `2 * batch_size * in_features * out_features` (computational complexity)
- Throughput is reported in GFLOPS (Giga Floating Point Operations Per Second) - execution speed
- Test case descriptions show computational workload in GFLOPs/TFLOPs/PFLOPs (operation count)

### Error Handling

- If a backend is not available, it's gracefully skipped with a warning
- All implementations include proper error handling and fallback mechanisms
- Test results include availability status for each backend

## Integration with Original Reference

The implementation is based on the `create_linear` function from:
`/home/jian.sha/atorch_addon/benchmarks/modules/fp8_linear/fp8_flops.py`

The operator extracts the FP8-specific linear module creation logic and integrates it into the op_bench framework with proper benchmarking infrastructure.

## Testing

Run the test suite:

```bash
python -m pytest tests/test_fp8_linear_operator.py -v
```

Or run individual tests:

```bash
python tests/test_fp8_linear_operator.py
```
