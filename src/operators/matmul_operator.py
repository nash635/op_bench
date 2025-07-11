#!/usr/bin/env python3
"""
MatMul Operator Implementation
Specific implementation of matrix multiplication operator
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase
import importlib.util

class MatMulOperator(BaseOperator):
    """Matrix multiplication operator implementation"""
    
    def __init__(self):
        super().__init__(OperatorType.MATMUL)
        self.matmul_cuda_ext = None  # Store the loaded CUDA extension
        self._setup_implementations()
        
    def get_test_cases(self) -> List[OperatorTestCase]:
        """Return test cases for matrix multiplication"""
        return [
            OperatorTestCase(
                name="small_square",
                input_shapes=[(256, 256), (256, 256)],
                input_dtypes=[torch.float32, torch.float32],
                description="Small square matrices"
            ),
            OperatorTestCase(
                name="medium_square",
                input_shapes=[(512, 512), (512, 512)],
                input_dtypes=[torch.float32, torch.float32],
                description="Medium square matrices"
            ),
            OperatorTestCase(
                name="large_square",
                input_shapes=[(1024, 1024), (1024, 1024)],
                input_dtypes=[torch.float32, torch.float32],
                description="Large square matrices"
            ),
            OperatorTestCase(
                name="rectangular",
                input_shapes=[(512, 1024), (1024, 256)],
                input_dtypes=[torch.float32, torch.float32],
                description="Rectangular matrices"
            ),
            OperatorTestCase(
                name="batch_matmul",
                input_shapes=[(4, 512, 512), (4, 512, 512)],
                input_dtypes=[torch.float32, torch.float32],
                description="Batch matrix multiplication"
            )
        ]
        
    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        """Generate input tensors for matrix multiplication"""
        inputs = []
        for i, (shape, dtype) in enumerate(zip(test_case.input_shapes, test_case.input_dtypes)):
            tensor = torch.randn(shape, dtype=dtype, device='cuda').contiguous()
            inputs.append(tensor)
        return inputs
        
    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        """Calculate FLOPs for matrix multiplication"""
        shape_a, shape_b = test_case.input_shapes[:2]
        
        if len(shape_a) == 2 and len(shape_b) == 2:
            # Standard matrix multiplication: A(m,k) × B(k,n) = C(m,n)
            # FLOPs = 2 * m * k * n (multiply-add operations)
            m, k = shape_a
            k2, n = shape_b
            assert k == k2, f"Matrix dimensions don't match: {k} vs {k2}"
            return 2 * m * k * n
        elif len(shape_a) == 3 and len(shape_b) == 3:
            # Batch matrix multiplication
            batch_size, m, k = shape_a
            batch_size2, k2, n = shape_b
            assert batch_size == batch_size2, f"Batch sizes don't match: {batch_size} vs {batch_size2}"
            assert k == k2, f"Matrix dimensions don't match: {k} vs {k2}"
            return batch_size * 2 * m * k * n
        else:
            raise ValueError(f"Unsupported shapes: {shape_a}, {shape_b}")
            
    def get_reference_result(self, inputs: List[torch.Tensor], 
                           test_case: OperatorTestCase) -> torch.Tensor:
        """Get reference result using PyTorch built-in"""
        if len(inputs[0].shape) == 2:
            return torch.mm(inputs[0], inputs[1])
        else:
            return torch.bmm(inputs[0], inputs[1])
            
    def _setup_implementations(self):
        """Setup all available implementations"""
        
        # PyTorch built-in implementations
        self.register_implementation(
            "pytorch_mm",
            self._pytorch_mm,
            "PyTorch torch.mm",
            "PyTorch built-in matrix multiplication"
        )
        
        self.register_implementation(
            "pytorch_matmul",
            self._pytorch_matmul,
            "PyTorch torch.matmul",
            "PyTorch unified matrix multiplication"
        )
        
        self.register_implementation(
            "pytorch_bmm",
            self._pytorch_bmm,
            "PyTorch torch.bmm",
            "PyTorch batch matrix multiplication"
        )
        
        # Try to load custom CUDA implementations
        cuda_extension_loaded = False
        try:
            # First try direct import
            import matmul_cuda_ext
            self.matmul_cuda_ext = matmul_cuda_ext
            cuda_extension_loaded = True
            print("[INFO] CUDA extension loaded successfully via direct import")
        except ImportError:
            try:
                # Try to load using torch.utils.cpp_extension.load
                from torch.utils.cpp_extension import load
                import os
                
                # Get the directory containing CUDA source files
                cuda_dir = os.path.join(os.path.dirname(__file__), '..', 'cuda')
                if os.path.exists(cuda_dir):
                    cuda_sources = [
                        os.path.join(cuda_dir, 'matmul_cuda_ext.cpp'),
                        os.path.join(cuda_dir, 'matmul_kernels.cu')
                    ]
                    
                    # Check if source files exist
                    if all(os.path.exists(src) for src in cuda_sources):
                        self.matmul_cuda_ext = load(
                            name='matmul_cuda_ext',
                            sources=cuda_sources,
                            extra_cflags=['-O3', '-std=c++17'],
                            extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
                            verbose=False
                        )
                        cuda_extension_loaded = True
                        print("[INFO] CUDA extension loaded successfully via torch.utils.cpp_extension.load")
                    else:
                        print(f"[WARN] CUDA source files not found in {cuda_dir}")
                else:
                    print(f"[WARN] CUDA directory not found: {cuda_dir}")
            except Exception as e:
                print(f"[WARN] Failed to load CUDA extension: {e}")
        
        if cuda_extension_loaded:
            self.register_implementation(
                "cuda_basic",
                self._cuda_basic,
                "Basic CUDA Kernel",
                "Basic CUDA matrix multiplication kernel"
            )
            
            self.register_implementation(
                "cuda_shared",
                self._cuda_shared,
                "CUDA Shared Memory",
                "CUDA kernel with shared memory optimization"
            )
            
            self.register_implementation(
                "cuda_static_shared",
                self._cuda_static_shared,
                "CUDA Static Shared Memory",
                "CUDA kernel with static shared memory"
            )
            
            self.register_implementation(
                "cuda_template_8",
                self._cuda_template_8,
                "CUDA Template (8x8)",
                "CUDA template kernel with 8x8 tile size"
            )
            
            self.register_implementation(
                "cuda_template_16",
                self._cuda_template_16,
                "CUDA Template (16x16)",
                "CUDA template kernel with 16x16 tile size"
            )
            
            self.register_implementation(
                "cuda_template_32",
                self._cuda_template_32,
                "CUDA Template (32x32)",
                "CUDA template kernel with 32x32 tile size"
            )
        else:
            print("[INFO] Framework running in compatibility mode (PyTorch + CuPy backends available)")
            
        # CuPy implementations
        try:
            import cupy as cp
            self.register_implementation(
                "cupy_dot",
                self._cupy_dot,
                "CuPy dot",
                "CuPy matrix multiplication using dot"
            )
            
            self.register_implementation(
                "cupy_matmul",
                self._cupy_matmul,
                "CuPy matmul",
                "CuPy matrix multiplication using matmul"
            )
            
        except ImportError:
            print("[WARN] CuPy not available")
            
        # Set reference implementation
        self.set_reference_implementation("pytorch_mm")
        
        # Add CPU/NumPy implementations for accuracy comparison
        self.register_implementation(
            "numpy_cpu",
            self._numpy_cpu,
            "NumPy CPU",
            "NumPy matrix multiplication on CPU"
        )
        
        self.register_implementation(
            "pytorch_cpu",
            self._pytorch_cpu,
            "PyTorch CPU",
            "PyTorch matrix multiplication on CPU"
        )
        
    def _pytorch_mm(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """PyTorch torch.mm implementation"""
        if len(inputs[0].shape) == 2:
            return torch.mm(inputs[0], inputs[1])
        else:
            return torch.bmm(inputs[0], inputs[1])
            
    def _pytorch_matmul(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """PyTorch torch.matmul implementation"""
        return torch.matmul(inputs[0], inputs[1])
        
    def _pytorch_bmm(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """PyTorch torch.bmm implementation"""
        if len(inputs[0].shape) == 3:
            return torch.bmm(inputs[0], inputs[1])
        else:
            # Convert 2D to 3D for bmm
            a_3d = inputs[0].unsqueeze(0)
            b_3d = inputs[1].unsqueeze(0)
            result_3d = torch.bmm(a_3d, b_3d)
            return result_3d.squeeze(0)
            
    def _cuda_basic(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA basic kernel"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_basic(inputs[0], inputs[1])
            return None
        except Exception as e:
            print(f"Error in cuda_basic: {e}")
            return None
            
    def _cuda_shared(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA shared memory kernel"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_shared(inputs[0], inputs[1])
            return None
        except Exception as e:
            print(f"Error in cuda_shared: {e}")
            return None
            
    def _cuda_static_shared(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA static shared memory kernel"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_static_shared(inputs[0], inputs[1])
            return None
        except Exception as e:
            print(f"Error in cuda_static_shared: {e}")
            return None
    
    def _cuda_template_8(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA template kernel with 8x8 tile size"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_template(inputs[0], inputs[1], 8)
            return None
        except Exception as e:
            print(f"Error in cuda_template_8: {e}")
            return None
    
    def _cuda_template_16(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA template kernel with 16x16 tile size"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_template(inputs[0], inputs[1], 16)
            return None
        except Exception as e:
            print(f"Error in cuda_template_16: {e}")
            return None
    
    def _cuda_template_32(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """Custom CUDA template kernel with 32x32 tile size"""
        try:
            if self.matmul_cuda_ext is not None:
                return self.matmul_cuda_ext.matmul_template(inputs[0], inputs[1], 32)
            return None
        except Exception as e:
            print(f"Error in cuda_template_32: {e}")
            return None
            
    def _cupy_dot(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """CuPy dot implementation"""
        try:
            import cupy as cp
            a_cp = cp.asarray(inputs[0].detach())
            b_cp = cp.asarray(inputs[1].detach())
            result_cp = cp.dot(a_cp, b_cp)
            return torch.as_tensor(result_cp, device='cuda')
        except:
            return None
            
    def _cupy_matmul(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """CuPy matmul implementation"""
        try:
            import cupy as cp
            a_cp = cp.asarray(inputs[0].detach())
            b_cp = cp.asarray(inputs[1].detach())
            result_cp = cp.matmul(a_cp, b_cp)
            return torch.as_tensor(result_cp, device='cuda')
        except:
            return None
        
    def _numpy_cpu(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """NumPy CPU implementation"""
        try:
            import numpy as np
            a_np = inputs[0].cpu().numpy()
            b_np = inputs[1].cpu().numpy()
            result_np = np.dot(a_np, b_np)
            return torch.as_tensor(result_np, device='cuda')
        except:
            return None
        
    def _pytorch_cpu(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        """PyTorch CPU implementation"""
        return torch.matmul(inputs[0].cpu(), inputs[1].cpu()).cuda()
