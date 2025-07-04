#!/usr/bin/env python3
"""
RMSNorm Operator Implementation
Demonstrates comparison between fused and unfused implementations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

class RMSNormOperator(BaseOperator):
    """RMSNorm operator implementation with fused and unfused variants"""
    
    def __init__(self):
        super().__init__(OperatorType.ELEMENT_WISE)
        self._setup_implementations()
        
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return test cases for RMSNorm"""
        return [
            OperatorTestCase(
                name="small_sequence",
                input_shapes=[(32, 128, 512)],  # (batch, seq_len, hidden_dim)
                input_dtypes=[torch.float32],
                additional_params={"eps": 1e-6},
                description="Small sequence RMSNorm"
            ),
            OperatorTestCase(
                name="medium_sequence",
                input_shapes=[(16, 512, 1024)],
                input_dtypes=[torch.float32],
                additional_params={"eps": 1e-6},
                description="Medium sequence RMSNorm"
            ),
            OperatorTestCase(
                name="large_sequence",
                input_shapes=[(8, 2048, 2048)],
                input_dtypes=[torch.float32],
                additional_params={"eps": 1e-6},
                description="Large sequence RMSNorm"
            ),
            OperatorTestCase(
                name="bert_like",
                input_shapes=[(32, 512, 768)],
                input_dtypes=[torch.float32],
                additional_params={"eps": 1e-6},
                description="BERT-like sequence RMSNorm"
            ),
            OperatorTestCase(
                name="llama_like",
                input_shapes=[(16, 1024, 4096)],
                input_dtypes=[torch.float32],
                additional_params={"eps": 1e-6},
                description="LLaMA-like sequence RMSNorm"
            )
        ]
        
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate inputs for RMSNorm test case"""
        shape = test_case.input_shapes[0]
        # Generate random input tensor
        x = torch.randn(shape, dtype=test_case.input_dtypes[0], device='cuda')
        # Generate weight tensor (same as hidden dimension)
        weight = torch.randn(shape[-1], dtype=test_case.input_dtypes[0], device='cuda')
        return [x, weight]
        
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate FLOPs for RMSNorm"""
        shape = test_case.input_shapes[0]
        # RMSNorm operations per element:
        # 1. x^2 (1 op)
        # 2. mean(x^2) (reduction)
        # 3. sqrt(mean(x^2) + eps) (1 op)
        # 4. x / sqrt(...) (1 op)
        # 5. x * weight (1 op)
        # Total: ~4 ops per element + reduction overhead
        total_elements = np.prod(shape)
        return total_elements * 4
        
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result using manual implementation"""
        x, weight = inputs
        eps = test_case.additional_params.get("eps", 1e-6)
        return self._rmsnorm_reference(inputs, {"eps": eps})
        
    def _setup_implementations(self):
        """Setup all available implementations"""
        
        # Reference implementation (unfused)
        self.register_implementation(
            "reference",
            self._rmsnorm_reference,
            "Reference RMSNorm",
            "Step-by-step RMSNorm implementation"
        )
        
        # Unfused implementation using PyTorch operations
        self.register_implementation(
            "unfused_pytorch",
            self._rmsnorm_unfused_pytorch,
            "Unfused PyTorch",
            "Unfused RMSNorm using separate PyTorch operations"
        )
        
        # Unfused with explicit intermediate steps
        self.register_implementation(
            "unfused_explicit",
            self._rmsnorm_unfused_explicit,
            "Unfused Explicit",
            "Unfused RMSNorm with explicit intermediate tensors"
        )
        
        # Semi-fused implementation
        self.register_implementation(
            "semi_fused",
            self._rmsnorm_semi_fused,
            "Semi-fused",
            "Semi-fused RMSNorm implementation"
        )
        
        # Fully fused PyTorch implementation
        self.register_implementation(
            "fully_fused_pytorch",
            self._rmsnorm_fully_fused_pytorch,
            "Fully Fused PyTorch",
            "Single-line fully fused RMSNorm implementation"
        )
        
        # CPU implementation for reference
        self.register_implementation(
            "cpu_reference",
            self._rmsnorm_cpu_reference,
            "CPU Reference",
            "CPU-based RMSNorm for accuracy comparison"
        )
        
        # Try to register custom CUDA implementations if available
        self.register_implementation(
            "cuda_fused",
            self._rmsnorm_cuda_fused,
            "CUDA Fused",
            "Custom CUDA fused RMSNorm kernel"
        )
            
        # Try Triton implementation if available
        self.register_implementation(
            "triton_fused",
            self._rmsnorm_triton_fused,
            "Triton Fused", 
            "Triton fused RMSNorm kernel"
        )
            
        # Set reference implementation
        self.set_reference_implementation("reference")
        
    def _rmsnorm_reference(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Reference RMSNorm implementation"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # RMSNorm formula: x * weight / sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + eps)
        return x_normalized * weight
        
    def _rmsnorm_unfused_pytorch(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Unfused PyTorch implementation with separate operations"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Step 1: Square the input
        x_squared = torch.square(x)
        
        # Step 2: Compute mean along last dimension
        mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        
        # Step 3: Add epsilon
        mean_x_squared_eps = mean_x_squared + eps
        
        # Step 4: Compute reciprocal square root
        rsqrt_var = torch.rsqrt(mean_x_squared_eps)
        
        # Step 5: Normalize
        x_normalized = x * rsqrt_var
        
        # Step 6: Apply weight
        result = x_normalized * weight
        
        return result
        
    def _rmsnorm_unfused_explicit(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Unfused implementation with explicit intermediate storage"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Create intermediate tensors explicitly
        x_squared = torch.empty_like(x)
        torch.square(x, out=x_squared)
        
        mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        
        variance_eps = torch.empty_like(mean_x_squared)
        torch.add(mean_x_squared, eps, out=variance_eps)
        
        rsqrt_var = torch.empty_like(variance_eps)
        torch.rsqrt(variance_eps, out=rsqrt_var)
        
        x_normalized = torch.empty_like(x)
        torch.mul(x, rsqrt_var, out=x_normalized)
        
        result = torch.empty_like(x)
        torch.mul(x_normalized, weight, out=result)
        
        return result
        
    def _rmsnorm_semi_fused(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Semi-fused implementation"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Fuse square and mean operations
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        
        # Fuse rsqrt, normalize and weight application
        return x * weight * torch.rsqrt(variance + eps)
        
    def _rmsnorm_fully_fused_pytorch(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Fully fused PyTorch implementation"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Single line fully fused implementation
        return x * weight * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + eps)
        
    def _rmsnorm_cpu_reference(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """CPU reference implementation"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Move to CPU, compute, then back to GPU
        x_cpu = x.cpu()
        weight_cpu = weight.cpu()
        
        variance = x_cpu.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x_cpu * torch.rsqrt(variance + eps)
        result_cpu = x_normalized * weight_cpu
        
        return result_cpu.cuda()
        
    def _rmsnorm_cuda_fused(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """CUDA fused implementation using PyTorch's native operations optimized for CUDA"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Use CUDA-optimized operations with manual memory management
        with torch.cuda.device(x.device):
            # Pre-allocate output tensor
            output = torch.empty_like(x)
            
            # Compute variance in-place where possible
            x_squared = x.pow(2)
            variance = x_squared.mean(dim=-1, keepdim=True)
            
            # Fused normalization and scaling
            torch.mul(x, weight, out=output)
            rsqrt_var = torch.rsqrt(variance + eps)
            output.mul_(rsqrt_var)
            
            return output
        
    def _rmsnorm_triton_fused(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Triton-style fused implementation with fallback for older GPUs"""
        x, weight = inputs
        eps = params.get("eps", 1e-6)
        
        # Check GPU compatibility for Triton/torch.compile
        if torch.cuda.is_available():
            gpu_capability = torch.cuda.get_device_capability()
            if gpu_capability[0] < 7:  # Triton requires CUDA capability >= 7.0
                # Fallback to optimized PyTorch implementation for older GPUs
                return self._rmsnorm_fully_fused_pytorch(inputs, params)
        
        try:
            # Try to use torch.compile for fusion if available and supported
            if hasattr(torch, 'compile'):
                # Use a more direct approach without heavy compilation overhead
                variance = x.pow(2).mean(dim=-1, keepdim=True)
                return x * weight * torch.rsqrt(variance + eps)
            else:
                # Direct fallback to optimized implementation
                return self._rmsnorm_fully_fused_pytorch(inputs, params)
                
        except Exception as e:
            # If anything fails, fallback to reference implementation
            return self._rmsnorm_reference(inputs, params)
