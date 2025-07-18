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
    # Set matplotlib cache directory to a writable location
    import os
    import tempfile
    if not os.path.exists(os.path.expanduser('~/.cache/matplotlib')):
        try:
            os.makedirs(os.path.expanduser('~/.cache/matplotlib'), exist_ok=True)
        except (OSError, PermissionError):
            # If we can't create the default cache directory, use a temp directory
            temp_dir = tempfile.mkdtemp(prefix='matplotlib_cache_')
            os.environ['MPLCONFIGDIR'] = temp_dir
    
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
    print("[INFO] Matplotlib available for chart generation")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] Matplotlib not available, skipping chart generation")
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"[WARN] Matplotlib not available: {e}, skipping chart generation")

from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase, ImplementationResult

class GPUProfiler:
    """Profile GPU capabilities and recommend optimal test configurations"""
    
    def __init__(self):
        self.device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        self.compute_capability = torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 0
        
    def get_gpu_class(self) -> str:
        """Determine GPU performance class"""
        device_name = self.device_name.upper()
        
        if 'B200' in device_name:
            return 'B200'
        elif 'H100' in device_name:
            return 'H100'
        elif 'A100' in device_name:
            return 'A100'
        elif 'V100' in device_name:
            return 'V100'
        elif 'RTX' in device_name and any(x in device_name for x in ['4090', '4080', '3090']):
            return 'HIGH_END_CONSUMER'
        elif 'GTX' in device_name or 'RTX' in device_name:
            return 'CONSUMER'
        else:
            return 'UNKNOWN'
    
    def get_theoretical_peak_flops(self) -> float:
        """Get theoretical peak FLOPS based on GPU"""
        gpu_class = self.get_gpu_class()
        
        peak_flops = {
            'B200': 2500e12,    # 2.5 PFLOPS FP8
            'H100': 1000e12,    # 1.0 PFLOPS FP8
            'A100': 312e12,     # 312 TFLOPS FP8
            'V100': 125e12,     # 125 TFLOPS FP16
            'HIGH_END_CONSUMER': 100e12,  # ~100 TFLOPS
            'CONSUMER': 50e12,  # ~50 TFLOPS
        }
        
        return peak_flops.get(gpu_class, 50e12)
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """Get adaptive configuration based on GPU capabilities"""
        gpu_class = self.get_gpu_class()
        memory_factor = min(self.total_memory_gb / 80.0, 1.0)  # Normalize to 80GB baseline
        
        config = {
            'device_name': self.device_name,
            'gpu_class': gpu_class,
            'memory_gb': self.total_memory_gb,
            'compute_capability': self.compute_capability,
            'theoretical_peak_flops': self.get_theoretical_peak_flops(),
            'memory_factor': memory_factor
        }
        
        # Recommend test cases based on GPU class
        if gpu_class in ['B200', 'H100']:
            test_cases = [
                'h100_target_small' if gpu_class == 'H100' else 'b200_target_medium',
                'h100_target_medium' if gpu_class == 'H100' else 'b200_target_large',
                'h100_target_large' if gpu_class == 'H100' else 'b200_peak_stress',
                # Add stress tests for high-memory configs
                f'{gpu_class.lower()}_peak_stress' if memory_factor >= 0.8 else None,
            ]
        elif gpu_class == 'A100':
            test_cases = ['medium_baseline', 'h100_target_small', 'h100_target_medium']
        else:
            test_cases = ['quick_validation', 'medium_baseline']
        
        config['recommended_test_cases'] = [tc for tc in test_cases if tc is not None]
        config['recommended_backends'] = ['pytorch_bf16']
        
        return config

class FP8BenchmarkConfigs:
    """Predefined benchmark configurations for different scenarios"""
    
    @staticmethod
    def get_h100_configs() -> Dict[str, Any]:
        """Benchmark configurations optimized for H100 GPU"""
        return {
            "h100_quick_validation": {
                "test_cases": ["quick_validation", "medium_baseline"],
                "backends": ["pytorch_bf16"],
                "runs": 3, "warmup": 2, "target_efficiency": 60
            },
            "h100_performance_suite": {
                "test_cases": ["h100_target_small", "h100_target_medium", "h100_target_large"],
                "backends": None, "runs": 5, "warmup": 3, "target_efficiency": 70
            },
            "h100_peak_stress": {
                "test_cases": ["h100_peak_stress"],
                "backends": ["pytorch_bf16"],
                "runs": 3, "warmup": 2, "target_efficiency": 80
            }
        }
    
    @staticmethod
    def get_b200_configs() -> Dict[str, Any]:
        """Benchmark configurations optimized for B200 GPU"""
        return {
            "b200_performance_suite": {
                "test_cases": ["b200_target_medium", "b200_target_large"],
                "backends": None, "runs": 5, "warmup": 3, "target_efficiency": 70
            },
            "b200_peak_stress": {
                "test_cases": ["b200_peak_stress"],
                "backends": ["pytorch_bf16"],
                "runs": 3, "warmup": 2, "target_efficiency": 85
            }
        }

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
            'relu': OperatorType.ACTIVATION,
            'rmsnorm': OperatorType.ELEMENT_WISE,
            'rdma_stress': OperatorType.RDMA_STRESS,
            'tcp_bandwidth': OperatorType.TCP_BANDWIDTH,
            'rdma_bandwidth': OperatorType.RDMA_BANDWIDTH,
            'pcie_bandwidth': OperatorType.PCIE_BANDWIDTH,
            'network_stress': OperatorType.NETWORK_STRESS,
            'fp8_linear': OperatorType.FP8_LINEAR
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
            'relu': OperatorType.ACTIVATION,
            'rmsnorm': OperatorType.ELEMENT_WISE,
            'rdma_stress': OperatorType.RDMA_STRESS,
            'tcp_bandwidth': OperatorType.TCP_BANDWIDTH,
            'rdma_bandwidth': OperatorType.RDMA_BANDWIDTH,
            'pcie_bandwidth': OperatorType.PCIE_BANDWIDTH,
            'network_stress': OperatorType.NETWORK_STRESS,
            'fp8_linear': OperatorType.FP8_LINEAR
        }
        
        if operator_type not in type_mapping:
            return []
            
        op_type = type_mapping[operator_type]
        if op_type in self.operators:
            test_cases = self.operators[op_type].get_test_cases()
            return [tc.name for tc in test_cases]
        return []
        
    def run(self, operator_type: str, implementations: List[str] = None, test_cases: List[str] = None,
                      warmup_runs: int = 5, test_runs: int = 20, 
                      collect_outputs: bool = False) -> Tuple[Dict[str, List[ImplementationResult]], Dict[str, Dict[str, torch.Tensor]]]:
        """Run comparison for a specific operator"""
        # Map string to OperatorType
        type_mapping = {
            'matmul': OperatorType.MATMUL,
            'vector_add': OperatorType.VECTOR_ADD,
            'relu': OperatorType.ACTIVATION,
            'rmsnorm': OperatorType.ELEMENT_WISE,
            'rdma_stress': OperatorType.RDMA_STRESS,
            'tcp_bandwidth': OperatorType.TCP_BANDWIDTH,
            'rdma_bandwidth': OperatorType.RDMA_BANDWIDTH,
            'pcie_bandwidth': OperatorType.PCIE_BANDWIDTH,
            'network_stress': OperatorType.NETWORK_STRESS,
            'fp8_linear': OperatorType.FP8_LINEAR
        }
        
        if operator_type not in type_mapping:
            raise ValueError(f"Operator {operator_type} not supported")
            
        op_type = type_mapping[operator_type]
        if op_type not in self.operators:
            raise ValueError(f"Operator {operator_type} not registered")
            
        operator = self.operators[op_type]
        all_test_cases = operator.get_test_cases()
        
        # Filter test cases
        if test_cases:
            filtered_test_cases = [tc for tc in all_test_cases if tc.name in test_cases]
        else:
            filtered_test_cases = all_test_cases
            
        results = {}
        output_results = {}
        
        for test_case in filtered_test_cases:
            print(f"\n=== Testing {operator_type.upper()} - {test_case.name} ===")
            print(f"Description: {test_case.description}")
            print(f"Input shapes: {test_case.input_shapes}")
            
            case_results = operator.run_comparison(
                test_case, implementations, warmup_runs, test_runs
            )
            
            # Collect outputs if requested
            if collect_outputs:
                output_results[test_case.name] = {}
                inputs = operator.generate_inputs(test_case)
                
                for impl_name in operator.implementations:
                    if implementations is None or impl_name in implementations:
                        try:
                            impl_info = operator.implementations[impl_name]
                            impl_func = impl_info['function']
                            with torch.no_grad():
                                output = impl_func(inputs, {})
                                output_results[test_case.name][impl_name] = output
                        except Exception as e:
                            print(f"  [WARN] Failed to collect output for {impl_name}: {e}")
                            output_results[test_case.name][impl_name] = None
            
            # Display results - Pure performance mode only
            for result in case_results:
                if result.available:
                    print(f"  [PERF] {result.name}: {result.avg_time_ms:.3f}ms, {result.display_metric}")
                else:
                    print(f"  [FAIL] {result.name}: Not available - {result.error}")
                    
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
            
            # Performance-only table headers
            is_network_test = any(result.is_network_test for result in case_results if result.available)
            
            if is_network_test:
                report += "| Implementation | Available | Avg Time (ms) | Bandwidth (Gbps) | Min Time (ms) | Std Dev (ms) |\n"
                report += "|----------------|-----------|---------------|------------------|---------------|---------------|\n"
            else:
                report += "| Implementation | Available | Avg Time (ms) | GFLOPS | Min Time (ms) | Std Dev (ms) |\n"
                report += "|----------------|-----------|---------------|--------|---------------|---------------|\n"
            
            for result in case_results:
                if result.available:
                    available_str = "PASS"
                    
                    if is_network_test:
                        metric_value = f"{result.bandwidth_gbps:.3f}"
                    else:
                        metric_value = f"{result.gflops:.1f}"
                    
                    # Performance-only row without correctness column
                    report += f"| {result.name} | {available_str} | {result.avg_time_ms:.3f} | {metric_value} | {result.min_time_ms:.3f} | {result.std_time_ms:.3f} |\n"
                else:
                    report += f"| {result.name} | FAIL | N/A | N/A | N/A | N/A |\n"
                    
            report += "\n"
            
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[INFO] Report saved: {output_file}")
            
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
                }
                
                # Add appropriate performance metric
                if result.is_network_test:
                    json_data['bandwidth_gbps'] = result.bandwidth_gbps
                else:
                    json_data['gflops'] = result.gflops
                
                if result.error:
                    json_data['error'] = result.error
                json_results[test_case_name].append(json_data)
                
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] JSON results saved: {output_file}")
        
    def create_performance_charts(self, results: Dict[str, List[ImplementationResult]], 
                                 operator_type: str, output_prefix: str = "comparison") -> List[str]:
        """Generate performance comparison charts"""
        if not MATPLOTLIB_AVAILABLE:
            print("[WARN] Matplotlib not available, skipping chart generation")
            return []
            
        chart_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare data for visualization
        test_cases = list(results.keys())
        implementations = set()
        for case_results in results.values():
            for result in case_results:
                # In performance mode, result.correct is None, so we check available only
                # In accuracy mode, we check both available and correct
                if result.available and (result.correct is True or result.correct is None):
                    implementations.add(result.name)
        implementations = sorted(list(implementations))
        
        # Check if we have any successful implementations
        if not implementations:
            print("[WARN] No successful implementations found for chart generation")
            return []
        
        # Check if this is a network test
        is_network_test = any(
            any(result.is_network_test for result in case_results 
                if result.available and (result.correct is True or result.correct is None))
            for case_results in results.values()
        )
        
        # Performance metric chart (GFLOPS for compute, Gbps for network)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(test_cases))
        # Optimize bar width for better visualization when there are few test cases
        if len(test_cases) == 1:
            # For single test case, use narrower bars and center them properly
            width = min(0.6 / len(implementations), 0.15)
            x_offset = 0.3  # Center the bars in the plot
        else:
            width = 0.8 / len(implementations)
            x_offset = 0
        
        for i, impl in enumerate(implementations):
            metric_data = []
            for test_case in test_cases:
                case_results = results[test_case]
                # Fix: Handle both performance mode (correct=None) and accuracy mode (correct=True/False)
                impl_result = next((r for r in case_results 
                                  if r.name == impl and r.available and (r.correct is True or r.correct is None)), None)
                if impl_result:
                    metric_value = impl_result.bandwidth_gbps if is_network_test else impl_result.gflops
                    metric_data.append(metric_value)
                else:
                    metric_data.append(0)
                
            bar_positions = x + i * width + x_offset
            ax.bar(bar_positions, metric_data, width, label=impl, alpha=0.8)
            
        ax.set_xlabel('Test Cases')
        
        if is_network_test:
            ax.set_ylabel('Bandwidth (Gbps)')
            ax.set_title(f'{operator_type.upper()} Performance Comparison (Bandwidth)')
            chart_filename = f"{output_prefix}_bandwidth_{timestamp}.png"
        else:
            ax.set_ylabel('GFLOPS')
            ax.set_title(f'{operator_type.upper()} Performance Comparison (GFLOPS)')
            chart_filename = f"{output_prefix}_gflops_{timestamp}.png"
        
        # Adjust x-axis settings based on number of test cases
        if len(test_cases) == 1:
            ax.set_xticks([0.3 + width * (len(implementations) - 1) / 2])
            ax.set_xlim(-0.2, 1.2)  # Limit x-axis range for single test case
        else:
            ax.set_xticks(x + width * (len(implementations) - 1) / 2)
            
        ax.set_xticklabels(test_cases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(chart_filename)
        
        # Execution time chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use the same width and offset settings as performance chart
        for i, impl in enumerate(implementations):
            times_data = []
            for test_case in test_cases:
                case_results = results[test_case]
                # Fix: Handle both performance mode (correct=None) and accuracy mode (correct=True/False)
                impl_result = next((r for r in case_results 
                                  if r.name == impl and r.available and (r.correct is True or r.correct is None)), None)
                times_data.append(impl_result.avg_time_ms if impl_result else 0)
                
            bar_positions = x + i * width + x_offset
            ax.bar(bar_positions, times_data, width, label=impl, alpha=0.8)
            
        ax.set_xlabel('Test Cases')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title(f'{operator_type.upper()} Performance Comparison (Execution Time)')
        
        # Adjust x-axis settings based on number of test cases
        if len(test_cases) == 1:
            ax.set_xticks([0.3 + width * (len(implementations) - 1) / 2])
            ax.set_xlim(-0.2, 1.2)  # Limit x-axis range for single test case
        else:
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
            print("[WARN] Matplotlib not available, skipping accuracy chart generation")
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
            print("[WARN] No accuracy data available for chart generation")
            return []
        
        # Create side-by-side subplots for relative and absolute error
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        x = np.arange(len(test_cases))
        # Optimize bar width for better visualization when there are few test cases
        if len(test_cases) == 1:
            # For single test case, use narrower bars and center them properly
            width = min(0.6 / len(implementations), 0.15)
            x_offset = 0.3  # Center the bars in the plot
        else:
            width = 0.8 / len(implementations)
            x_offset = 0
        
        # Plot 1: Relative Error
        for i, impl in enumerate(implementations):
            relative_errors = []
            for test_case in test_cases:
                if test_case in accuracy_metrics and impl in accuracy_metrics[test_case]:
                    relative_errors.append(accuracy_metrics[test_case][impl]['relative_error'])
                else:
                    relative_errors.append(0)
            
            bar_positions = x + i * width + x_offset
            ax1.bar(bar_positions, relative_errors, width, label=impl, alpha=0.8)
        
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Relative Error')
        ax1.set_title(f'{operator_type.upper()} Relative Error vs PyTorch CPU')
        
        # Adjust x-axis settings based on number of test cases
        if len(test_cases) == 1:
            ax1.set_xticks([0.3 + width * (len(implementations) - 1) / 2])
            ax1.set_xlim(-0.2, 1.2)  # Limit x-axis range for single test case
        else:
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
            
            bar_positions = x + i * width + x_offset
            ax2.bar(bar_positions, mae_values, width, label=impl, alpha=0.8)
        
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title(f'{operator_type.upper()} Mean Absolute Error vs PyTorch CPU')
        
        # Adjust x-axis settings based on number of test cases
        if len(test_cases) == 1:
            ax2.set_xticks([0.3 + width * (len(implementations) - 1) / 2])
            ax2.set_xlim(-0.2, 1.2)  # Limit x-axis range for single test case
        else:
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
    parser.add_argument('--operator', choices=['matmul', 'vector_add', 'relu', 'rmsnorm', 'rdma_stress', 
                                               'tcp_bandwidth', 'rdma_bandwidth', 'pcie_bandwidth', 'network_stress', 'fp8_linear'],
                       help='Operator type to test')
    parser.add_argument('--test-cases', nargs='+', 
                       help='Test cases to run (default: all)')
    parser.add_argument('--implementations', nargs='+',
                       help='Implementations to test (default: all)')
    parser.add_argument('--output', default='comparison',
                       help='Output file prefix')
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory (default: test_results)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup rounds')
    parser.add_argument('--runs', type=int, default=20,
                       help='Test rounds')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance comparison charts')
    parser.add_argument('--output-diff', action='store_true',
                       help='Generate accuracy difference analysis (using CPU as reference)')
    parser.add_argument('--accuracy-only', action='store_true',
                       help='Run only accuracy comparison (no performance testing)')
    parser.add_argument('--list-operators', action='store_true',
                       help='List all available operators')
    parser.add_argument('--list-implementations', action='store_true',
                       help='List implementations for the specified operator')
    parser.add_argument('--list-test-cases', action='store_true',
                       help='List test cases for the specified operator')
    
    # FP8 Linear adaptive mode options
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive mode - automatically select optimal test cases and backends based on GPU')
    parser.add_argument('--stress-tests', action='store_true',
                       help='Include stress tests (requires high-memory GPU)')
    parser.add_argument('--profile-only', action='store_true',
                       help='Profile GPU capabilities only (no benchmarking)')
    parser.add_argument('--config', choices=['h100_quick_validation', 'h100_performance_suite', 'h100_peak_stress',
                                            'b200_performance_suite', 'b200_peak_stress'],
                       help='Use predefined configuration for specific GPU class')
    
    args = parser.parse_args()
    
    # Initialize comparator and register operators
    comparator = UniversalOperatorComparator()
    
    # Profile GPU if adaptive mode or profile-only is requested
    if args.adaptive or args.profile_only or args.config:
        gpu_profiler = GPUProfiler()
        gpu_config = gpu_profiler.get_adaptive_config()
        
        print("=" * 60)
        print("GPU CAPABILITY ANALYSIS")
        print("=" * 60)
        print(f"Device: {gpu_config['device_name']}")
        print(f"GPU Class: {gpu_config['gpu_class']}")
        print(f"Memory: {gpu_config['memory_gb']:.1f} GB")
        print(f"Compute Capability: {gpu_config['compute_capability']}")
        print(f"Theoretical Peak FP8: {gpu_config['theoretical_peak_flops']/1e12:.1f} TFLOPS")
        print(f"Memory Factor: {gpu_config['memory_factor']:.2f}")
        
        if args.profile_only:
            print("\nRECOMMENDED CONFIGURATION:")
            print(f"Test Cases: {', '.join(gpu_config['recommended_test_cases'][:3])}")
            print(f"Backends: {', '.join(gpu_config['recommended_backends'])}")
            return 0
    
    # Handle predefined configurations
    if args.config:
        configs = FP8BenchmarkConfigs()
        if args.config.startswith('h100'):
            config_dict = configs.get_h100_configs()[args.config]
        elif args.config.startswith('b200'):
            config_dict = configs.get_b200_configs()[args.config]
        else:
            print(f"[ERROR] Unknown configuration: {args.config}")
            return 1
        
        # Override args with config values
        args.operator = 'fp8_linear'
        args.test_cases = config_dict['test_cases']
        args.implementations = config_dict['backends']
        args.runs = config_dict['runs']
        args.warmup = config_dict['warmup']
        args.plot = True
        
        print(f"\nUSING PREDEFINED CONFIG: {args.config}")
        print(f"Test Cases: {', '.join(args.test_cases)}")
        print(f"Expected Efficiency: {config_dict['target_efficiency']}%")
    
    # Handle adaptive mode
    elif args.adaptive:
        if not torch.cuda.is_available():
            print("[ERROR] Adaptive mode requires CUDA")
            return 1
            
        # Use GPU profiler recommendations
        args.operator = 'fp8_linear'
        args.test_cases = gpu_config['recommended_test_cases']
        args.implementations = gpu_config['recommended_backends']
        args.plot = True
        
        # Add stress tests if requested and GPU supports it
        if args.stress_tests and gpu_config['memory_factor'] >= 0.8:
            stress_test = f"{gpu_config['gpu_class'].lower()}_peak_stress"
            if stress_test not in args.test_cases:
                args.test_cases.append(stress_test)
        
        print(f"\nADAPTIVE MODE FOR {gpu_config['gpu_class']}")
        print(f"Selected Test Cases: {', '.join(args.test_cases)}")
        print(f"Selected Backends: {', '.join(args.implementations)}")

    args = parser.parse_args()
    
    # Create comparator and register operators
    comparator = UniversalOperatorComparator()
    
    # Register available operators
    try:
        from operators.matmul_operator import MatMulOperator
        comparator.register_operator(MatMulOperator())
    except ImportError as e:
        print(f"[WARN] MatMul operator not available: {e}")
        
    try:
        from operators.vector_add_operator import VectorAddOperator
        comparator.register_operator(VectorAddOperator())
    except ImportError as e:
        print(f"[WARN] VectorAdd operator not available: {e}")
        
    try:
        from operators.relu_operator import ReLUOperator
        comparator.register_operator(ReLUOperator())
    except ImportError as e:
        print(f"[WARN] ReLU operator not available: {e}")
        
    try:
        from operators.rmsnorm_operator import RMSNormOperator
        comparator.register_operator(RMSNormOperator())
    except ImportError as e:
        print(f"[WARN] RMSNorm operator not available: {e}")
    
    try:
        from operators.rdma_stress_operator import RDMAStressOperator
        comparator.register_operator(RDMAStressOperator())
    except ImportError as e:
        print(f"[WARN] RDMA Stress operator not available: {e}")
    
    # Register network performance operators
    try:
        from operators.tcp_bandwidth_operator import TCPBandwidthOperator
        comparator.register_operator(TCPBandwidthOperator())
    except ImportError as e:
        print(f"[WARN] TCP Bandwidth operator not available: {e}")
    
    try:
        from operators.rdma_bandwidth_operator import RDMABandwidthOperator
        comparator.register_operator(RDMABandwidthOperator())
    except ImportError as e:
        print(f"[WARN] RDMA Bandwidth operator not available: {e}")
    
    try:
        from operators.pcie_bandwidth_operator import PCIeBandwidthOperator
        comparator.register_operator(PCIeBandwidthOperator())
    except ImportError as e:
        print(f"[WARN] PCIe Bandwidth operator not available: {e}")
    
    try:
        from operators.network_stress_operator import NetworkStressOperator
        comparator.register_operator(NetworkStressOperator())
    except ImportError as e:
        print(f"[WARN] Network Stress operator not available: {e}")
    
    try:
        from operators.fp8_linear_operator import FP8LinearOperator
        comparator.register_operator(FP8LinearOperator())
    except ImportError as e:
        print(f"[WARN] FP8 Linear operator not available: {e}")
    
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
        print("[FAIL] CUDA not available")
        return 1
        
    print("=" * 60)
    print(f"{args.operator.upper()} Implementation Comparison Test")
    print("=" * 60)
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    if args.accuracy_only:
        print("Running accuracy comparison only...")
        
        # Map operator name to type
        type_mapping = {
            'matmul': OperatorType.MATMUL,
            'vector_add': OperatorType.VECTOR_ADD,
            'relu': OperatorType.ACTIVATION,
            'rmsnorm': OperatorType.ELEMENT_WISE,
            'rdma_stress': OperatorType.RDMA_STRESS,
            'tcp_bandwidth': OperatorType.TCP_BANDWIDTH,
            'rdma_bandwidth': OperatorType.RDMA_BANDWIDTH,
            'pcie_bandwidth': OperatorType.PCIE_BANDWIDTH,
            'network_stress': OperatorType.NETWORK_STRESS,
            'fp8_linear': OperatorType.FP8_LINEAR
        }
        
        if args.operator not in type_mapping:
            print(f"[FAIL] Unknown operator: {args.operator}")
            return 1
            
        op_type = type_mapping[args.operator]
        if op_type not in comparator.operators:
            print(f"[FAIL] Operator {args.operator} not registered")
            return 1
            
        operator = comparator.operators[op_type]
        test_cases = operator.get_test_cases()
        if args.test_cases:
            test_cases = [tc for tc in test_cases if tc.name in args.test_cases]
        
        for test_case in test_cases:
            print(f"\n[INFO] Running accuracy comparison for test case: {test_case.name}")
            
            accuracy_results = operator.run_accuracy_comparison(test_case, args.implementations)
            
            print(f"\nAccuracy Results for {test_case.name}:")
            print("-" * 60)
            
            # Extract and display baseline info if available
            baseline_info = accuracy_results.pop('_baseline_info', None)
            if baseline_info:
                print(f"Baseline (Reference): {baseline_info['reference_display_name']} ({baseline_info['reference_impl_id']})")
                print("-" * 60)
            
            for impl_id, metrics in accuracy_results.items():
                status = metrics.get('status', 'UNKNOWN')
                if status == 'PASS':
                    print(f"{impl_id}: PASS (tolerance: {metrics.get('passed_tolerance', 'N/A')})")
                elif status == 'FAIL':
                    print(f"{impl_id}: FAIL (max_error: {metrics.get('max_error', 'N/A'):.2e})")
                else:
                    print(f"? {impl_id}: {status} ({metrics.get('error', 'Unknown error')})")
        
        return 0
    
    results, output_results = comparator.run(
        args.operator, args.implementations, args.test_cases,
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
            if 'cpu_reference' in case_outputs and case_outputs['cpu_reference'] is not None:
                reference_result = case_outputs['cpu_reference']
            elif 'pytorch_cpu' in case_outputs and case_outputs['pytorch_cpu'] is not None:
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
            print(f"[INFO] Accuracy metrics saved: {accuracy_json_file}")
        else:
            print("[WARN] No suitable reference implementation found for accuracy comparison")
    
    print("=" * 60)
    print("Comparison test completed!")
    print(f"[INFO] Detailed report: {report_file}")
    print(f"[INFO] JSON data: {json_file}")
    if chart_files:
        print("[INFO] Performance charts:")
        for chart_file in chart_files:
            print(f"    [CHART] {chart_file}")
    if accuracy_chart_files:
        print("[INFO] Accuracy charts:")
        for chart_file in accuracy_chart_files:
            print(f"    [CHART] {chart_file}")
    
    return 0

if __name__ == "__main__":
    main()
