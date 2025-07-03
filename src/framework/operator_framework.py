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
    correct: bool
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    gflops: float
    additional_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[torch.Tensor] = None

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
            
    def verify_correctness(self, result: torch.Tensor, reference: torch.Tensor, 
                         tolerance: float = 1e-4) -> bool:
        """Verify if result matches reference"""
        if result is None or reference is None:
            return False
        try:
            return torch.allclose(result, reference, atol=tolerance, rtol=tolerance)
        except Exception:
            return False
            
    def benchmark_implementation(self, impl_id: str, inputs: List[torch.Tensor],
                               test_case: OperatorTestCase, warmup_runs: int = 5,
                               test_runs: int = 20) -> ImplementationResult:
        """Benchmark a specific implementation"""
        if impl_id not in self.implementations:
            return ImplementationResult(
                name=impl_id,
                impl_id=impl_id,
                available=False,
                correct=False,
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
                    correct=False,
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
            
            # Check correctness
            reference = self.get_reference_result(inputs, test_case)
            is_correct = self.verify_correctness(result, reference)
            
            return ImplementationResult(
                name=display_name,
                impl_id=impl_id,
                available=True,
                correct=is_correct,
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
                correct=False,
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
        """Run comparison for a specific test case"""
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
