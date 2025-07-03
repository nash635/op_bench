# Universal Operator Framework Guide

## Overview

This framework provides a flexible system for implementing and comparing different operator implementations across various backends (PyTorch, CUDA, CuPy, etc.).

## Framework Components

### 1. Base Framework (`operator_framework.py`)
- `BaseOperator`: Abstract base class for all operators
- `OperatorType`: Enum defining supported operator types
- `OperatorTestCase`: Configuration for test cases
- `ImplementationResult`: Result structure for benchmarks

### 2. Operator Implementations
- `MatMulOperator`: Matrix multiplication implementations
- `VectorAddOperator`: Vector addition implementations
- (Extensible for new operators)

### 3. Universal Comparator (`operator_comparator_tool.py`)
- Command-line tool for running comparisons
- Visualization and reporting capabilities
- Support for multiple operators

## Adding New Operators

### Step 1: Define the Operator Class

```python
from operator_framework import BaseOperator, OperatorType, OperatorTestCase

class YourOperator(BaseOperator):
    def __init__(self):
        super().__init__(OperatorType.YOUR_OPERATOR)
        self._setup_implementations()
        
    def get_test_cases(self) -> List[OperatorTestCase]:
        return [
            OperatorTestCase(
                name="test_case_1",
                input_shapes=[(1024, 1024)],
                input_dtypes=[torch.float32],
                description="Description of test case"
            )
        ]
        
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        # Generate test inputs based on test case
        pass
        
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        # Calculate FLOPs for the operation
        pass
        
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        # Provide reference implementation for correctness checking
        pass
```

### Step 2: Register Implementations

```python
def _setup_implementations(self):
    self.register_implementation(
        "impl_id",
        self._implementation_function,
        "Display Name",
        "Description"
    )
    
def _implementation_function(self, inputs: List[torch.Tensor], 
                           params: Dict[str, Any]) -> torch.Tensor:
    # Your implementation here
    pass
```

### Step 3: Update the Universal Comparator

Add your operator to the `operator_comparator_tool.py`:

```python
# Add to choices in argument parser
parser.add_argument('--operator', choices=['matmul', 'vector_add', 'your_operator'])

# Add registration in main()
try:
    from your_operator import YourOperator
    comparator.register_operator(YourOperator())
except ImportError as e:
    print(f"⚠️  YourOperator not available: {e}")
```

## Usage Examples

### Basic Usage

```bash
# List available operators
python operator_comparator_tool.py --list-operators

# List implementations for an operator
python operator_comparator_tool.py --list-implementations matmul

# List test cases for an operator
python operator_comparator_tool.py --list-test-cases matmul

# Run comparison for MatMul
python operator_comparator_tool.py --operator matmul --plot

# Run comparison for VectorAdd
python operator_comparator_tool.py --operator vector_add --plot
```

### Advanced Usage

```bash
# Test specific implementations
python operator_comparator_tool.py \
    --operator matmul \
    --implementations pytorch_mm cuda_basic cuda_shared \
    --plot

# Test specific test cases
python operator_comparator_tool.py \
    --operator matmul \
    --test-cases small_square medium_square \
    --plot

# Custom output directory
python operator_comparator_tool.py \
    --operator matmul \
    --output-dir results/ \
    --output matmul_benchmark \
    --plot
```

## Framework Benefits

### 1. Modularity
- Each operator is self-contained
- Easy to add new operators
- Clean separation of concerns

### 2. Flexibility
- Support for different input types and shapes
- Configurable test cases
- Multiple implementation backends

### 3. Consistency
- Unified interface for all operators
- Consistent benchmarking methodology
- Standardized output formats

### 4. Extensibility
- Easy to add new implementations
- Support for custom metrics
- Pluggable visualization

## Implementation Examples

### MatMul Operator
- PyTorch: `torch.mm`, `torch.matmul`, `torch.bmm`
- CUDA: Custom kernels with shared memory optimization
- CuPy: `cp.dot`, `cp.matmul`

### VectorAdd Operator
- PyTorch: `torch.add`, `+` operator
- CUDA: Custom vector addition kernels
- CuPy: `cp.add`

## Output Formats

### 1. Markdown Report
- Detailed performance comparison tables
- Test case descriptions
- Implementation availability status

### 2. JSON Data
- Machine-readable results
- All performance metrics
- Error information

### 3. Performance Charts
- GFLOPS comparison charts
- Execution time charts
- Professional visualization

## Best Practices

### 1. Operator Implementation
- Always provide a reference implementation
- Handle edge cases gracefully
- Use appropriate error handling

### 2. Test Cases
- Cover various input sizes
- Include edge cases
- Provide meaningful descriptions

### 3. Implementation Functions
- Return None for unavailable implementations
- Handle exceptions properly
- Ensure tensor contiguity when needed

### 4. Performance Considerations
- Use proper warmup rounds
- Synchronize CUDA operations
- Account for memory transfer overhead

## Migration from Existing Code

### From matmul_comparator.py
1. Extract operator-specific logic to `MatMulOperator`
2. Move implementations to the operator class
3. Use the universal comparator for CLI

### Benefits of Migration
- Reusable code across operators
- Consistent interface
- Easier maintenance
- Better extensibility

## Future Extensions

### Potential New Operators
- Convolution operators
- Activation functions (ReLU, GELU, etc.)
- Reduction operations (sum, mean, etc.)
- Element-wise operations
- Custom domain-specific operators

### Enhanced Features
- Multi-GPU support
- Memory usage tracking
- Energy consumption metrics
- Automatic optimization suggestions
