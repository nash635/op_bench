# FP8 Linear Operator Documentation

## Overview

The FP8 Linear Operator provides high-performance linear layer implementations optimized for modern GPUs, particularly H100 and B200 class accelerators. It supports multiple backends and includes comprehensive test cases designed to maximize computational throughput.

## Features

- Multiple FP8 backend implementations (PyTorch BF16, Transformer Engine, CUTLASS, cuBLAS)
- High-performance test cases optimized for H100/B200 GPUs
- Real-world LLM workload scenarios (LLaMA3-405B, GPT-4 scale)
- Memory bandwidth optimization tests
- Automatic GPU capability detection and adaptive benchmarking
- CHART Automatic chart generation with --plot flag
- TARGET Pre-configured test scenarios (quick test, full analysis, etc.)
- IDEA Analysis tips and guidance

## Test Cases

### Quick Validation
- `quick_validation`: 256×512 → 1024 (0.27 GFLOPS) - Basic functionality test
- `medium_baseline`: 1024×2048 → 4096 (16.78 GFLOPS) - Medium-scale baseline

### H100 Optimized Test Cases
- `h100_target_small`: 8192×8192 → 8192 (1.07 TFLOPS) - H100 small target
- `h100_target_medium`: 16384×12288 → 16384 (6.44 TFLOPS) - H100 medium target  
- `h100_target_large`: 32768×16384 → 32768 (35.18 TFLOPS) - H100 large target
- `h100_peak_stress`: 65536×24576 → 49152 (157.29 TFLOPS) - H100 peak stress test

### B200 Optimized Test Cases
- `b200_target_medium`: 65536×32768 → 65536 (274.88 TFLOPS) - B200 medium target
- `b200_target_large`: 131072×49152 → 98304 (1.23 PFLOPS) - B200 large target
- `b200_peak_stress`: 262144×65536 → 131072 (4.40 PFLOPS) - B200 ultimate stress test

### Real-World LLM Workloads
- `llama3_405b_ffn_up`: 32768×16384 → 53248 (35.18 TFLOPS) - LLaMA3-405B FFN up projection
- `llama3_405b_ffn_down`: 32768×53248 → 16384 (57.27 TFLOPS) - LLaMA3-405B FFN down projection
- `gpt4_scale_attention`: 131072×32768 → 98304 (824.63 TFLOPS) - GPT-4 scale attention

### Memory Bandwidth Tests
- `memory_bandwidth_square`: 32768×32768 → 32768 (70.37 TFLOPS) - Square matrix optimization
- `memory_bandwidth_wide`: 16384×131072 → 16384 (70.37 TFLOPS) - Wide matrix test
- `memory_bandwidth_tall`: 131072×16384 → 16384 (70.37 TFLOPS) - Tall matrix test

## Performance Targets

### H100 SXM (1000 TFLOPS FP8 theoretical peak)
- Target efficiency: 60-80% of theoretical peak
- Optimal test cases: `h100_target_*` series
- Peak stress test: `h100_peak_stress` (157+ TFLOPS)

### B200 (2500 TFLOPS FP8 theoretical peak)  
- Target efficiency: 70-85% of theoretical peak
- Optimal test cases: `b200_target_*` series
- Peak stress test: `b200_peak_stress` (4+ PFLOPS)

# Run FP8 linear benchmarks with all available backends and generate charts
python run_comparator.py --operator fp8_linear --warmup 5 --runs 10 --plot

# Run specific test cases with plotting
python run_comparator.py --operator fp8_linear --test-cases small_linear medium_linear --plot

# Run with specific implementations
python run_comparator.py --operator fp8_linear --implementations pytorch_bf16 simulated_fp8_fast --plot

# Generate charts and save results
python run_comparator.py --operator fp8_linear --plot --save-results results.json
```

#### Multi-Backend Performance Comparison

The FP8 Linear operator now supports multiple backends for comparison:

1. **pytorch_bf16**: PyTorch BF16 baseline (always available)
2. **simulated_fp8_fast**: Simulated high-performance FP8 implementation
3. **simulated_fp8_memory_opt**: Simulated memory-optimized FP8 implementation
4. **transformer_engine**: NVIDIA Transformer Engine FP8 (if installed)
5. **cutlass/cublas**: Custom FP8 implementations (if available)

#### Chart Generation

When using `--plot`, the tool generates:
- **Performance GFLOPS Chart**: Compares computational throughput across backends
- **Execution Time Chart**: Compares execution times across different problem sizes
- **Detailed Reports**: Markdown and JSON formats with comprehensive results

### Available Test Cases

1. **small_linear**: 256x512 -> 1024 linear layer
2. **medium_linear**: 512x1024 -> 2048 linear layer  
3. **large_linear**: 1024x2048 -> 4096 linear layer
4. **transformer_like**: 2048x4096 -> 6144 linear layer
5. **llm_ffn**: 4096x6144 -> 13312 linear layer (LLM feed-forward network)

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

# Run benchmarks
results, outputs = comparator.run(
    operator_type="fp8_linear",
    implementations=["pytorch_bf16", "transformer_engine"],
    test_cases=["medium_linear"],
    warmup_runs=5,
    test_runs=10
)

# Print results
for test_case, impl_results in results.items():
    print(f"\nTest Case: {test_case}")
    for result in impl_results:
        print(f"  {result.impl_id}: {result.display_metric}")
```

## Implementation Details

### Backend Detection

The operator automatically detects available backends:

- **pytorch_bf16**: Always available (baseline)
- **transformer_engine**: Available if `transformer_engine` package is installed
- **cutlass/cublas**: Available if custom `linear` module is importable

### Memory and Performance

- All implementations use bfloat16 input data type
- Input tensors are generated with small random values (scaled by 0.1) for FP8 compatibility
- FLOP calculations: `2 * batch_size * in_features * out_features`
- Throughput is reported in GFLOPS (Giga Floating Point Operations Per Second)

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
