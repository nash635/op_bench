#!/usr/bin/env python3
"""
Universal Operator Comparator Tool
A flexible tool for comparing different operator implementations
"""

import torch
import time
import numpy as np
import argparse
from typing import List, Dict, Tuple, Any, Optional
import os
import sys
import json
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available for chart generation")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available, skipping chart generation")

from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase, ImplementationResult

class UniversalOperatorComparator:
    """Universal comparator for all operator types"""
    
    def __init__(self):
        self.operators = {}
        self.results_cache = {}
        
    def register_operator(self, operator: BaseOperator):
        """Register an operator for comparison"""
        self.operators[operator.operator_type] = operator
        
    def list_operators(self) -> List[str]:
        """List all registered operators"""
        return [op_type.value for op_type in self.operators.keys()]
        
    def list_implementations(self, operator_type: str) -> List[str]:
        """List implementations for a specific operator"""
        # Map string to OperatorType
        type_mapping = {
            'matmul': OperatorType.MATMUL,
            'vector_add': OperatorType.VECTOR_ADD,
            'relu': OperatorType.ACTIVATION
        }
        
        if operator_type not in type_mapping:
            return []
            
        op_type = type_mapping[operator_type]
        if op_type in self.operators:
            return list(self.operators[op_type].implementations.keys())
        return []
        
    def list_test_cases(self, operator_type: str) -> List[str]:
        """List test cases for a specific operator"""
        # Map string to OperatorType
        type_mapping = {
            'matmul': OperatorType.MATMUL,
            'vector_add': OperatorType.VECTOR_ADD,
            'relu': OperatorType.ACTIVATION
        }
        
        if operator_type not in type_mapping:
            return []
            
        op_type = type_mapping[operator_type]
        if op_type in self.operators:
            test_cases = self.operators[op_type].get_test_cases()
            return [tc.name for tc in test_cases]
        return []
        
    def run_comparison(self, operator_type: str, test_case_names: Optional[List[str]] = None,
                      implementation_names: Optional[List[str]] = None,
                      warmup_runs: int = 5, test_runs: int = 20, 
                      collect_outputs: bool = False) -> Tuple[Dict[str, List[ImplementationResult]], Dict[str, Dict[str, torch.Tensor]]]:
        """Run comparison for a specific operator"""
        # Map string to OperatorType
        type_mapping = {
            'matmul': OperatorType.MATMUL,
            'vector_add': OperatorType.VECTOR_ADD,
            'relu': OperatorType.ACTIVATION
        }
        
        if operator_type not in type_mapping:
            raise ValueError(f"Operator {operator_type} not supported")
            
        op_type = type_mapping[operator_type]
        if op_type not in self.operators:
            raise ValueError(f"Operator {operator_type} not registered")
            
        operator = self.operators[op_type]
        test_cases = operator.get_test_cases()
        
        # Filter test cases
        if test_case_names:
            test_cases = [tc for tc in test_cases if tc.name in test_case_names]
            
        results = {}
        output_results = {}
        
        for test_case in test_cases:
            print(f"\n=== Testing {operator_type.upper()} - {test_case.name} ===")
            print(f"Description: {test_case.description}")
            print(f"Input shapes: {test_case.input_shapes}")
            
            case_results = operator.run_comparison(
                test_case, implementation_names, warmup_runs, test_runs
            )
            
            # Collect outputs if requested
            if collect_outputs:
                output_results[test_case.name] = {}
                inputs = operator.generate_inputs(test_case)
                
                for impl_name in operator.implementations:
                    if implementation_names is None or impl_name in implementation_names:
                        try:
                            impl_info = operator.implementations[impl_name]
                            impl_func = impl_info['function']
                            with torch.no_grad():
                                output = impl_func(inputs, {})
                                output_results[test_case.name][impl_name] = output
                        except Exception as e:
                            print(f"  ⚠️  Failed to collect output for {impl_name}: {e}")
                            output_results[test_case.name][impl_name] = None
            
            # Display results
            for result in case_results:
                if result.available and result.correct:
                    print(f"  ✅ {result.name}: {result.avg_time_ms:.3f}ms, {result.gflops:.1f} GFLOPS")
                elif result.available and not result.correct:
                    print(f"  ❌ {result.name}: {result.avg_time_ms:.3f}ms, {result.gflops:.1f} GFLOPS (INCORRECT)")
                else:
                    print(f"  ❌ {result.name}: Not available - {result.error}")
                    
            results[test_case.name] = case_results
            
        return results, output_results
        
    def generate_report(self, results: Dict[str, List[ImplementationResult]], 
                       operator_type: str, output_file: str = None) -> str:
        """Generate comparison report"""
        report = f"""# {operator_type.upper()} Implementation Comparison Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Device**: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}

## Performance Results

"""
        
        for test_case_name, case_results in results.items():
            report += f"\n### Test Case: {test_case_name}\n\n"
            report += "| Implementation | Available | Correct | Avg Time (ms) | GFLOPS | Min Time (ms) | Std Dev (ms) |\n"
            report += "|----------------|-----------|---------|---------------|--------|---------------|---------------|\n"
            
            for result in case_results:
                if result.available:
                    available_str = "✅" if result.correct else "⚠️"
                    correct_str = "✅" if result.correct else "❌"
                    report += f"| {result.name} | {available_str} | {correct_str} | {result.avg_time_ms:.3f} | {result.gflops:.1f} | {result.min_time_ms:.3f} | {result.std_time_ms:.3f} |\n"
                else:
                    report += f"| {result.name} | ❌ | N/A | N/A | N/A | N/A | N/A |\n"
                    
            report += "\n"
            
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved: {output_file}")
            
        return report
        
    def save_json_results(self, results: Dict[str, List[ImplementationResult]], 
                         output_file: str):
        """Save JSON format results"""
        json_results = {}
        
        for test_case_name, case_results in results.items():
            json_results[test_case_name] = []
            for result in case_results:
                json_data = {
                    'name': result.name,
                    'impl_id': result.impl_id,
                    'available': result.available,
                    'correct': result.correct,
                    'avg_time_ms': result.avg_time_ms,
                    'std_time_ms': result.std_time_ms,
                    'min_time_ms': result.min_time_ms,
                    'max_time_ms': result.max_time_ms,
                    'gflops': result.gflops
                }
                if result.error:
                    json_data['error'] = result.error
                json_results[test_case_name].append(json_data)
                
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON results saved: {output_file}")
        
    def create_performance_charts(self, results: Dict[str, List[ImplementationResult]], 
                                 operator_type: str, output_prefix: str = "comparison") -> List[str]:
        """Generate performance comparison charts"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping chart generation")
            return []
            
        chart_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare data for visualization
        test_cases = list(results.keys())
        implementations = set()
        for case_results in results.values():
            for result in case_results:
                if result.available and result.correct:
                    implementations.add(result.name)
        implementations = sorted(list(implementations))
        
        # Check if we have any successful implementations
        if not implementations:
            print("⚠️  No successful implementations found for chart generation")
            return []
        
        # GFLOPS chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(test_cases))
        width = 0.8 / len(implementations)
        
        for i, impl in enumerate(implementations):
            gflops_data = []
            for test_case in test_cases:
                case_results = results[test_case]
                impl_result = next((r for r in case_results if r.name == impl and r.available and r.correct), None)
                gflops_data.append(impl_result.gflops if impl_result else 0)
                
            ax.bar(x + i * width, gflops_data, width, label=impl, alpha=0.8)
            
        ax.set_xlabel('Test Cases')
        ax.set_ylabel('GFLOPS')
        ax.set_title(f'{operator_type.upper()} Performance Comparison (GFLOPS)')
        ax.set_xticks(x + width * (len(implementations) - 1) / 2)
        ax.set_xticklabels(test_cases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        gflops_file = f"{output_prefix}_gflops_{timestamp}.png"
        plt.savefig(gflops_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(gflops_file)
        
        # Execution time chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, impl in enumerate(implementations):
            times_data = []
            for test_case in test_cases:
                case_results = results[test_case]
                impl_result = next((r for r in case_results if r.name == impl and r.available and r.correct), None)
                times_data.append(impl_result.avg_time_ms if impl_result else 0)
                
            ax.bar(x + i * width, times_data, width, label=impl, alpha=0.8)
            
        ax.set_xlabel('Test Cases')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title(f'{operator_type.upper()} Performance Comparison (Execution Time)')
        ax.set_xticks(x + width * (len(implementations) - 1) / 2)
        ax.set_xticklabels(test_cases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        times_file = f"{output_prefix}_times_{timestamp}.png"
        plt.savefig(times_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(times_file)
        
        return chart_files
    
    def calculate_accuracy_metrics(self, reference_results: Dict[str, torch.Tensor],
                                  test_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate accuracy metrics comparing all implementations to reference"""
        accuracy_metrics = {}
        
        for test_case_name, ref_result in reference_results.items():
            accuracy_metrics[test_case_name] = {}
            
            if test_case_name in test_results:
                for impl_name, impl_result in test_results[test_case_name].items():
                    if impl_result is not None:
                        # Convert to CPU numpy for comparison
                        ref_np = ref_result.detach().cpu().numpy()
                        impl_np = impl_result.detach().cpu().numpy()
                        
                        # Calculate various accuracy metrics
                        mse = np.mean((ref_np - impl_np) ** 2)
                        mae = np.mean(np.abs(ref_np - impl_np))
                        max_abs_error = np.max(np.abs(ref_np - impl_np))
                        
                        # Relative error (avoid division by zero)
                        ref_norm = np.linalg.norm(ref_np)
                        if ref_norm > 0:
                            relative_error = np.linalg.norm(ref_np - impl_np) / ref_norm
                        else:
                            relative_error = 0.0
                        
                        # Cosine similarity
                        ref_flat = ref_np.flatten()
                        impl_flat = impl_np.flatten()
                        cosine_sim = np.dot(ref_flat, impl_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(impl_flat))
                        
                        accuracy_metrics[test_case_name][impl_name] = {
                            'mse': float(mse),
                            'mae': float(mae),
                            'max_abs_error': float(max_abs_error),
                            'relative_error': float(relative_error),
                            'cosine_similarity': float(cosine_sim)
                        }
        
        return accuracy_metrics
    
    def create_accuracy_charts(self, accuracy_metrics: Dict[str, Dict[str, Dict[str, float]]],
                             operator_type: str, output_prefix: str = "accuracy") -> List[str]:
        """Generate accuracy comparison charts with relative and absolute error bars"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping accuracy chart generation")
            return []
        
        chart_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare data for visualization
        test_cases = list(accuracy_metrics.keys())
        implementations = set()
        for case_metrics in accuracy_metrics.values():
            implementations.update(case_metrics.keys())
        
        # Remove reference implementation from comparison (since it would be 0)
        implementations.discard('pytorch_cpu')
        implementations.discard('numpy_cpu')
        implementations = sorted(list(implementations))
        
        if not implementations:
            print("⚠️  No accuracy data available for chart generation")
            return []
        
        # Create side-by-side subplots for relative and absolute error
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        x = np.arange(len(test_cases))
        width = 0.8 / len(implementations)
        
        # Plot 1: Relative Error
        for i, impl in enumerate(implementations):
            relative_errors = []
            for test_case in test_cases:
                if test_case in accuracy_metrics and impl in accuracy_metrics[test_case]:
                    relative_errors.append(accuracy_metrics[test_case][impl]['relative_error'])
                else:
                    relative_errors.append(0)
            
            ax1.bar(x + i * width, relative_errors, width, label=impl, alpha=0.8)
        
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Relative Error')
        ax1.set_title(f'{operator_type.upper()} Relative Error vs PyTorch CPU')
        ax1.set_xticks(x + width * (len(implementations) - 1) / 2)
        ax1.set_xticklabels(test_cases, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Mean Absolute Error
        for i, impl in enumerate(implementations):
            mae_values = []
            for test_case in test_cases:
                if test_case in accuracy_metrics and impl in accuracy_metrics[test_case]:
                    mae_values.append(accuracy_metrics[test_case][impl]['mae'])
                else:
                    mae_values.append(0)
            
            ax2.bar(x + i * width, mae_values, width, label=impl, alpha=0.8)
        
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title(f'{operator_type.upper()} Mean Absolute Error vs PyTorch CPU')
        ax2.set_xticks(x + width * (len(implementations) - 1) / 2)
        ax2.set_xticklabels(test_cases, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        accuracy_file = f"{output_prefix}_accuracy_{timestamp}.png"
        plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(accuracy_file)
        
        return chart_files
def main():
    parser = argparse.ArgumentParser(description='Universal Operator Comparison Tool')
    parser.add_argument('--operator', choices=['matmul', 'vector_add', 'relu'],
                       help='Operator type to test')
    parser.add_argument('--test-cases', nargs='+', 
                       help='Test cases to run (default: all)')
    parser.add_argument('--implementations', nargs='+',
                       help='Implementations to test (default: all)')
    parser.add_argument('--output', default='comparison',
                       help='Output file prefix')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup rounds')
    parser.add_argument('--runs', type=int, default=20,
                       help='Test rounds')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance comparison charts')
    parser.add_argument('--output-diff', action='store_true',
                       help='Generate accuracy difference analysis (using CPU as reference)')
    parser.add_argument('--list-operators', action='store_true',
                       help='List all available operators')
    parser.add_argument('--list-implementations', action='store_true',
                       help='List implementations for the specified operator')
    parser.add_argument('--list-test-cases', action='store_true',
                       help='List test cases for the specified operator')
    
    args = parser.parse_args()
    
    # Create comparator and register operators
    comparator = UniversalOperatorComparator()
    
    # Register available operators
    try:
        from operators.matmul_operator import MatMulOperator
        comparator.register_operator(MatMulOperator())
    except ImportError as e:
        print(f"⚠️  MatMul operator not available: {e}")
        
    try:
        from operators.vector_add_operator import VectorAddOperator
        comparator.register_operator(VectorAddOperator())
    except ImportError as e:
        print(f"⚠️  VectorAdd operator not available: {e}")
        
    try:
        from operators.relu_operator import ReLUOperator
        comparator.register_operator(ReLUOperator())
    except ImportError as e:
        print(f"⚠️  ReLU operator not available: {e}")
    
    # Handle list commands first
    if args.list_operators:
        print("Available operators:")
        for op in comparator.list_operators():
            print(f"  - {op}")
        return 0
        
    if args.list_implementations:
        if not args.operator:
            parser.error("--operator is required when listing implementations")
        impls = comparator.list_implementations(args.operator)
        print(f"Available implementations for {args.operator}:")
        for impl in impls:
            print(f"  - {impl}")
        return 0
        
    if args.list_test_cases:
        if not args.operator:
            parser.error("--operator is required when listing test cases")
        test_cases = comparator.list_test_cases(args.operator)
        print(f"Available test cases for {args.operator}:")
        for tc in test_cases:
            print(f"  - {tc}")
        return 0
    
    # Require operator for non-list commands
    if not args.operator:
        parser.error("--operator is required for comparison tests")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
        
    print("=" * 60)
    print(f"{args.operator.upper()} Implementation Comparison Test")
    print("=" * 60)
    print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    results, output_results = comparator.run_comparison(
        args.operator, args.test_cases, args.implementations,
        args.warmup, args.runs, collect_outputs=args.output_diff
    )
    
    # Generate outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save report
    report_file = os.path.join(args.output_dir, f"{args.output}_{timestamp}.md")
    comparator.generate_report(results, args.operator, report_file)
    
    # Save JSON
    json_file = os.path.join(args.output_dir, f"{args.output}_{timestamp}.json")
    comparator.save_json_results(results, json_file)
    
    # Generate charts
    chart_files = []
    if args.plot:
        chart_prefix = os.path.join(args.output_dir, args.output)
        chart_files = comparator.create_performance_charts(results, args.operator, chart_prefix)
    
    # Generate accuracy analysis if requested
    accuracy_chart_files = []
    if args.output_diff and output_results:
        # Find PyTorch CPU implementation as reference
        reference_results = {}
        
        for test_case_name, case_outputs in output_results.items():
            reference_result = None
            # Use PyTorch CPU as reference
            if 'pytorch_cpu' in case_outputs and case_outputs['pytorch_cpu'] is not None:
                reference_result = case_outputs['pytorch_cpu']
            elif 'numpy_cpu' in case_outputs and case_outputs['numpy_cpu'] is not None:
                reference_result = case_outputs['numpy_cpu']
            
            if reference_result is not None:
                reference_results[test_case_name] = reference_result
        
        if reference_results:
            accuracy_metrics = comparator.calculate_accuracy_metrics(reference_results, output_results)
            accuracy_prefix = os.path.join(args.output_dir, f"{args.output}_accuracy")
            accuracy_chart_files = comparator.create_accuracy_charts(accuracy_metrics, args.operator, accuracy_prefix)
            
            # Save accuracy metrics to JSON
            accuracy_json_file = os.path.join(args.output_dir, f"{args.output}_accuracy_{timestamp}.json")
            with open(accuracy_json_file, 'w', encoding='utf-8') as f:
                json.dump(accuracy_metrics, f, indent=2, ensure_ascii=False)
            print(f"✅ Accuracy metrics saved: {accuracy_json_file}")
    
    print("=" * 60)
    print("Comparison test completed!")
    print(f"📄 Detailed report: {report_file}")
    print(f"📊 JSON data: {json_file}")
    if chart_files:
        print("📈 Performance charts:")
        for chart_file in chart_files:
            print(f"    📊 {chart_file}")
    if accuracy_chart_files:
        print("📈 Accuracy charts:")
        for chart_file in accuracy_chart_files:
            print(f"    📊 {chart_file}")
    
    return 0

if __name__ == "__main__":
    main()
