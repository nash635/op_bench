#!/usr/bin/env python3
"""
Universal Operator Framework
A flexible framework for implementing and comparing different operator implementations
"""

import torch
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import importlib.util
import os

class OperatorType(Enum):
    """Supported operator types"""
    MATMUL = "matmul"
    VECTOR_ADD = "vector_add"
    CONVOLUTION = "convolution"
    ACTIVATION = "activation"
    REDUCTION = "reduction"
    ELEMENT_WISE = "element_wise"
    RDMA_STRESS = "rdma_stress"
    TCP_BANDWIDTH = "tcp_bandwidth"
    RDMA_BANDWIDTH = "rdma_bandwidth"
    PCIE_BANDWIDTH = "pcie_bandwidth"
    NETWORK_STRESS = "network_stress"
    FP8_LINEAR = "fp8_linear"

@dataclass
class OperatorTestCase:
    """Test case configuration for an operator"""
    name: str
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[torch.dtype]
    additional_params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

@dataclass
class ImplementationResult:
    """Result of running an implementation"""
    name: str
    impl_id: str
    available: bool
    correct: Optional[bool]  # Make correctness optional
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    gflops: float
    additional_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[torch.Tensor] = None
    
    @property
    def is_network_test(self) -> bool:
        """Check if this is a network performance test"""
        return any(keyword in self.impl_id.lower() for keyword in ['tcp', 'rdma', 'pcie', 'bandwidth', 'network'])
    
    @property
    def bandwidth_gbps(self) -> float:
        """Get bandwidth in Gbps for network tests"""
        if self.is_network_test and self.result is not None:
            try:
                # For network tests, the result tensor contains bandwidth in Gbps
                return float(self.result.item())
            except:
                return 0.0
        return 0.0
    
    @property
    def display_metric(self) -> str:
        """Get the appropriate display metric string"""
        if self.is_network_test:
            return f"{self.bandwidth_gbps:.3f} Gbps"
        else:
            return f"{self.gflops:.1f} GFLOPS"

class BaseOperator(ABC):
    """Base class for all operators"""
    
    def __init__(self, operator_type: OperatorType):
        self.operator_type = operator_type
        self.implementations = {}
        self.reference_impl = None
        
    @abstractmethod
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return list of test cases for this operator"""
        pass
        
    @abstractmethod
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate input tensors for a test case"""
        pass
        
    @abstractmethod
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate FLOPs for a test case"""
        pass
        
    @abstractmethod
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result for correctness verification"""
        pass
        
    def register_implementation(self, impl_id: str, impl_func: callable, 
                              display_name: str, description: str = ""):
        """Register an implementation"""
        self.implementations[impl_id] = {
            'function': impl_func,
            'display_name': display_name,
            'description': description
        }
        
    def set_reference_implementation(self, impl_id: str):
        """Set the reference implementation for correctness checking"""
        if impl_id in self.implementations:
            self.reference_impl = impl_id
        else:
            raise ValueError(f"Implementation {impl_id} not found")
            
    def get_reference_implementation(self) -> str:
        """Get the reference implementation ID"""
        if self.reference_impl is None:
            # 如果没有设置reference，返回第一个可用的实现
            if self.implementations:
                return list(self.implementations.keys())[0]
            else:
                raise ValueError("No implementations available")
        return self.reference_impl
    
    def has_reference_implementation(self) -> bool:
        """Check if a reference implementation is set"""
        return self.reference_impl is not None and self.reference_impl in self.implementations
    
    def verify_correctness(self, result: torch.Tensor, reference: torch.Tensor, 
                         tolerance: float = None) -> bool:
        """Verify if result matches reference"""
        if result is None or reference is None:
            return False
        try:
            # Auto-adjust tolerance based on GPU architecture if not specified
            if tolerance is None:
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name()
                    # Higher-end GPUs (A100, H100, GB200) may have slight numerical differences
                    if any(gpu in device_name for gpu in ['A100', 'H100', 'GB200', 'V100']):
                        tolerance = 1e-3  # More relaxed for high-end GPUs
                    else:
                        tolerance = 1e-4  # Standard tolerance
                else:
                    tolerance = 1e-4
            
            return torch.allclose(result, reference, atol=tolerance, rtol=tolerance)
        except Exception:
            return False
            
    def benchmark_implementation(self, impl_id: str, inputs: List[torch.Tensor],
                               test_case: OperatorTestCase, warmup_runs: int = 5,
                               test_runs: int = 20) -> ImplementationResult:
        """Benchmark a specific implementation
        
        Args:
            impl_id: Implementation identifier
            inputs: Input tensors
            test_case: Test case configuration
            warmup_runs: Number of warmup iterations
            test_runs: Number of benchmark iterations
        """
        if impl_id not in self.implementations:
            return ImplementationResult(
                name=impl_id,
                impl_id=impl_id,
                available=False,
                correct=None,
                avg_time_ms=0.0,
                std_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                gflops=0.0,
                error="Implementation not found"
            )
            
        impl_info = self.implementations[impl_id]
        impl_func = impl_info['function']
        display_name = impl_info['display_name']
        
        try:
            # Test if implementation is available
            test_result = impl_func(inputs, test_case.additional_params or {})
            if test_result is None:
                return ImplementationResult(
                    name=display_name,
                    impl_id=impl_id,
                    available=False,
                    correct=None,
                    avg_time_ms=0.0,
                    std_time_ms=0.0,
                    min_time_ms=0.0,
                    max_time_ms=0.0,
                    gflops=0.0,
                    error="Implementation returned None"
                )
                
            # Warmup
            for _ in range(warmup_runs):
                _ = impl_func(inputs, test_case.additional_params or {})
                
            # Benchmark
            torch.cuda.synchronize()
            times = []
            for _ in range(test_runs):
                start_time = time.time()
                result = impl_func(inputs, test_case.additional_params or {})
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
            # Calculate statistics
            avg_time_ms = np.mean(times)
            std_time_ms = np.std(times)
            min_time_ms = np.min(times)
            max_time_ms = np.max(times)
            
            # Calculate GFLOPS
            flops = self.calculate_flops(test_case)
            gflops = flops / (avg_time_ms * 1e6)  # Convert to GFLOPS
            
            # Pure performance mode - no correctness checking
            return ImplementationResult(
                name=display_name,
                impl_id=impl_id,
                available=True,
                correct=None,  # Always None for performance mode
                avg_time_ms=avg_time_ms,
                std_time_ms=std_time_ms,
                min_time_ms=min_time_ms,
                max_time_ms=max_time_ms,
                gflops=gflops,
                result=result
            )
            
        except Exception as e:
            return ImplementationResult(
                name=display_name,
                impl_id=impl_id,
                available=False,
                correct=None,
                avg_time_ms=0.0,
                std_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                gflops=0.0,
                error=str(e)
            )
            
    def run_comparison(self, test_case: OperatorTestCase, 
                      selected_impls: Optional[List[str]] = None,
                      warmup_runs: int = 5, test_runs: int = 20) -> List[ImplementationResult]:
        """Run comparison for a specific test case
        
        Args:
            test_case: Test case configuration
            selected_impls: List of implementation IDs to test (None for all)
            warmup_runs: Number of warmup iterations
            test_runs: Number of benchmark iterations
        """
        if selected_impls is None:
            selected_impls = list(self.implementations.keys())
            
        inputs = self.generate_inputs(test_case)
        results = []
        
        for impl_id in selected_impls:
            if impl_id in self.implementations:
                result = self.benchmark_implementation(
                    impl_id, inputs, test_case, warmup_runs, test_runs
                )
                results.append(result)
                
        return results
    
    def run_implementation(self, impl_id: str, inputs: List[torch.Tensor], 
                          params: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Run a specific implementation with given inputs and parameters"""
        if impl_id not in self.implementations:
            raise ValueError(f"Implementation '{impl_id}' not found. Available: {list(self.implementations.keys())}")
        
        impl_func = self.implementations[impl_id]['function']
        if params is None:
            params = {}
        
        return impl_func(inputs, params)
    
    def get_available_implementations(self) -> List[str]:
        """Get list of available implementation IDs"""
        return list(self.implementations.keys())
    
    def get_implementation_info(self, impl_id: str) -> Dict[str, str]:
        """Get information about a specific implementation"""
        if impl_id not in self.implementations:
            raise ValueError(f"Implementation '{impl_id}' not found")
        return {
            'id': impl_id,
            'display_name': self.implementations[impl_id]['display_name'],
            'description': self.implementations[impl_id]['description']
        }
    
    def run_accuracy_comparison(self, test_case: OperatorTestCase, 
                              selected_impls: Optional[List[str]] = None,
                              reference_impl: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run accuracy comparison for implementations
        
        Args:
            test_case: Test case configuration
            selected_impls: List of implementation IDs to test (None for all)
            reference_impl: Reference implementation ID (None for auto-detection)
            
        Returns:
            Dictionary with accuracy metrics for each implementation, plus baseline info
        """
        if selected_impls is None:
            selected_impls = list(self.implementations.keys())
            
        if reference_impl is None:
            reference_impl = self.get_reference_implementation()
            
        inputs = self.generate_inputs(test_case)
        
        # Get reference result
        try:
            reference_result = self.get_reference_result(inputs, test_case)
        except Exception as e:
            print(f"[ERROR] Failed to get reference result: {e}")
            return {}
        
        accuracy_results = {}
        
        # Add baseline information at the top
        ref_info = self.get_implementation_info(reference_impl)
        accuracy_results['_baseline_info'] = {
            'reference_impl_id': reference_impl,
            'reference_display_name': ref_info['display_name'],
            'reference_description': ref_info['description']
        }
        
        for impl_id in selected_impls:
            if impl_id not in self.implementations:
                continue
                
            try:
                impl_func = self.implementations[impl_id]['function']
                result = impl_func(inputs, test_case.additional_params or {})
                
                if result is None:
                    accuracy_results[impl_id] = {
                        'status': 'FAIL',
                        'error': 'Implementation returned None',
                        'max_error': float('inf'),
                        'mean_error': float('inf'),
                        'relative_error': float('inf')
                    }
                    continue
                
                # Calculate error metrics
                abs_error = torch.abs(result - reference_result)
                max_error = abs_error.max().item()
                mean_error = abs_error.mean().item()
                relative_error = (abs_error / (torch.abs(reference_result) + 1e-8)).max().item()
                
                # Test different tolerance levels
                tolerances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
                passed_tolerance = None
                for tol in tolerances:
                    if self.verify_correctness(result, reference_result, tolerance=tol):
                        passed_tolerance = tol
                        break
                
                status = 'PASS' if passed_tolerance is not None else 'FAIL'
                
                accuracy_results[impl_id] = {
                    'status': status,
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'relative_error': relative_error,
                    'passed_tolerance': passed_tolerance,
                    'has_nan': torch.isnan(result).any().item(),
                    'has_inf': torch.isinf(result).any().item()
                }
                
            except Exception as e:
                accuracy_results[impl_id] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'max_error': float('inf'),
                    'mean_error': float('inf'),
                    'relative_error': float('inf')
                }
        
        return accuracy_results
