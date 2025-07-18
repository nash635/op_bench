#!/usr/bin/env python3
"""
FP8 Linear Layer Implementation using cuBLAS
Custom implementation for FP8 linear operations with cuBLAS backend
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

class Fp8Linear(nn.Module):
    """
    FP8 Linear layer implementation using cuBLAS backend
    
    This implementation provides FP8 linear operations for both CUTLASS and cuBLAS backends.
    It supports torch.float8_e4m3fn and torch.float8_e5m2 dtypes for efficient computation.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 backend: str = "cublas"):
        """
        Initialize FP8 Linear layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features  
            bias: Whether to include bias term
            device: Device to place the layer on
            dtype: Data type for weights (FP8 or fallback)
            backend: Backend to use ('cutlass' or 'cublas')
        """
        super(Fp8Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine appropriate dtype
        if dtype is None or not self._is_fp8_dtype(dtype):
            # Default to FP8 if available, otherwise fallback to bfloat16
            if hasattr(torch, 'float8_e4m3fn'):
                self.weight_dtype = torch.float8_e4m3fn
            else:
                self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = dtype
            
        print(f"INFO [Fp8Linear]: Creating {backend} FP8 linear layer")
        print(f"INFO [Fp8Linear]: {in_features} -> {out_features}, dtype={self.weight_dtype}, device={self.device}")
        
        # Initialize weights
        self._init_weights(bias)
        
        # FP8 scaling factors for quantization
        self.input_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.weight_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.output_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
        
    def _is_fp8_dtype(self, dtype: torch.dtype) -> bool:
        """Check if dtype is an FP8 type"""
        return "float8" in str(dtype)
        
    def _init_weights(self, bias: bool):
        """Initialize weight and bias parameters"""
        # Initialize weights with appropriate dtype
        if self._is_fp8_dtype(self.weight_dtype):
            # For FP8, initialize as float32 then convert
            weight_init = torch.randn(self.out_features, self.in_features, 
                                    dtype=torch.float32, device=self.device) * 0.02
            # Clamp to FP8 range and convert
            weight_init = torch.clamp(weight_init, -0.5, 0.5)
            self.weight = nn.Parameter(weight_init.to(self.weight_dtype))
        else:
            # Standard initialization for non-FP8
            self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features,
                                                 dtype=self.weight_dtype, device=self.device) * 0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, 
                                               dtype=torch.float32, device=self.device))
        else:
            self.register_parameter('bias', None)
            
    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input to FP8 if needed"""
        if not self._is_fp8_dtype(x.dtype) and self._is_fp8_dtype(self.weight_dtype):
            # Convert input to FP8 with scaling
            scaled_x = x * self.input_scale
            # Clamp to prevent overflow
            clamped_x = torch.clamp(scaled_x, -0.5, 0.5)
            return clamped_x.to(self.weight_dtype)
        return x
        
    def _dequantize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Dequantize output from FP8 if needed"""
        if self._is_fp8_dtype(y.dtype):
            # Convert back to float32/bfloat16 with scaling
            return (y.to(torch.float32) / self.output_scale).to(torch.bfloat16)
        return y
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FP8 linear layer
        
        Args:
            input: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is on correct device
        if input.device != self.device:
            input = input.to(self.device)
            
        # Quantize input if using FP8
        quantized_input = self._quantize_input(input)
        
        # Perform matrix multiplication
        if self.backend == "cublas":
            output = self._cublas_gemm(quantized_input, self.weight)
        elif self.backend == "cutlass":
            output = self._cutlass_gemm(quantized_input, self.weight)
        else:
            # Fallback to standard PyTorch
            output = torch.nn.functional.linear(quantized_input, self.weight, self.bias)
            
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        # Dequantize output
        output = self._dequantize_output(output)
        
        return output
        
    def _cublas_gemm(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Perform GEMM using cuBLAS backend
        
        For now, this falls back to PyTorch's implementation which uses cuBLAS internally.
        In a production implementation, this would call cuBLAS directly for optimal FP8 performance.
        """
        # Note: PyTorch's linear function already uses cuBLAS when available
        # For true FP8 cuBLAS, we would need to call cuBLAS APIs directly
        
        if self._is_fp8_dtype(input.dtype) and self._is_fp8_dtype(weight.dtype):
            # For FP8, convert to float32 for computation, then back to FP8
            input_f32 = input.to(torch.float32)
            weight_f32 = weight.to(torch.float32)
            output_f32 = torch.nn.functional.linear(input_f32, weight_f32, None)
            return output_f32.to(input.dtype)
        else:
            return torch.nn.functional.linear(input, weight, None)
            
    def _cutlass_gemm(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Perform GEMM using CUTLASS backend
        
        For now, this falls back to PyTorch's implementation.
        In a production implementation, this would use CUTLASS kernels for optimal FP8 performance.
        """
        # Similar to cuBLAS implementation for now
        return self._cublas_gemm(input, weight)
        
    def extra_repr(self) -> str:
        """String representation of the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, dtype={self.weight_dtype}, backend={self.backend}'

# Convenience function for creating FP8 linear layers
def create_fp8_linear(in_features: int, 
                      out_features: int, 
                      backend: str = "cublas",
                      bias: bool = False,
                      device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None) -> Fp8Linear:
    """
    Create an FP8 linear layer with specified backend
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        backend: Backend to use ('cutlass' or 'cublas')
        bias: Whether to include bias
        device: Device to place layer on
        dtype: Weight dtype (will auto-select FP8 if None)
        
    Returns:
        Fp8Linear layer instance
    """
    return Fp8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        dtype=dtype,
        backend=backend
    )

# Test function to verify FP8 linear layer functionality
def test_fp8_linear():
    """Test FP8 linear layer with different backends"""
    print("=== Testing FP8 Linear Layer ===")
    
    # Test parameters
    batch_size = 256
    in_features = 512
    out_features = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Test config: {batch_size}x{in_features} -> {out_features} on {device}")
    
    # Create test input
    input_tensor = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device=device) * 0.1
    print(f"Input: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
    
    # Test both backends
    for backend in ["cublas", "cutlass"]:
        print(f"\n--- Testing {backend} backend ---")
        try:
            # Create layer
            layer = create_fp8_linear(
                in_features=in_features,
                out_features=out_features,
                backend=backend,
                device=device
            )
            
            # Forward pass
            with torch.no_grad():
                output = layer(input_tensor)
                
            print(f"SUCCESS {backend}: Output shape={output.shape}, dtype={output.dtype}")
            print(f"SUCCESS {backend}: Output range=[{output.min():.4f}, {output.max():.4f}]")
            
        except Exception as e:
            print(f"ERROR {backend}: {e}")
    
    print("\n=== FP8 Linear Layer Test Complete ===")

if __name__ == "__main__":
    test_fp8_linear()
