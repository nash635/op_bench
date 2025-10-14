#!/usr/bin/env python3
"""Histogram Operator Implementation"""

import torch
from typing import List, Dict, Any
from functools import reduce
import operator
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from framework.operator_framework import BaseOperator, OperatorType, OperatorTestCase

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _histogram_triton_kernel(input_ptr, histogram_ptr, N, num_bins, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0)
        valid = mask & (values >= 0) & (values < num_bins)
        tl.atomic_add(histogram_ptr + values, 1, mask=valid)


def _is_triton_supported() -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 7


class HistogramOperator(BaseOperator):
    """Histogram operator leveraging custom CUDA kernel"""

    def __init__(self):
        super().__init__(OperatorType.HISTOGRAM)
        self.histogram_cuda_ext = None
        self._setup_implementations()

    def get_test_cases(self) -> List[OperatorTestCase]:
        return [
            OperatorTestCase(
                name="tiny",
                input_shapes=[(1024,)],
                input_dtypes=[torch.int32],
                additional_params={"num_bins": 32},
                description="Small histogram with few bins"
            ),
            OperatorTestCase(
                name="medium_dense",
                input_shapes=[(1 << 20,)],
                input_dtypes=[torch.int32],
                additional_params={"num_bins": 256},
                description="Medium input with dense distribution"
            ),
            OperatorTestCase(
                name="large_sparse",
                input_shapes=[(4 << 20,)],
                input_dtypes=[torch.int32],
                additional_params={"num_bins": 1024},
                description="Large input with many bins"
            )
        ]

    def generate_inputs(self, test_case: OperatorTestCase) -> List[torch.Tensor]:
        num_bins = test_case.additional_params["num_bins"]
        shape = test_case.input_shapes[0]
        tensor = torch.randint(
            low=0,
            high=num_bins,
            size=shape,
            dtype=torch.int32,
            device="cuda"
        ).contiguous()
        return [tensor]

    def calculate_flops(self, test_case: OperatorTestCase) -> int:
        return reduce(operator.mul, test_case.input_shapes[0], 1)

    def get_reference_result(self, inputs: List[torch.Tensor], test_case: OperatorTestCase) -> torch.Tensor:
        num_bins = test_case.additional_params["num_bins"]
        data = inputs[0]
        device = data.device
        try:
            result = torch.bincount(data.to(torch.int64), minlength=num_bins)
            if result.device != device:
                result = result.to(device)
        except RuntimeError:
            result = torch.bincount(data.cpu().to(torch.int64), minlength=num_bins).to(device)
        return result.to(torch.int32)

    def _setup_implementations(self):
        self.register_implementation(
            "pytorch_bincount",
            self._pytorch_bincount,
            "PyTorch bincount",
            "PyTorch torch.bincount on CUDA"
        )

        self.register_implementation(
            "cpu_bincount",
            self._cpu_bincount,
            "CPU bincount",
            "CPU torch.bincount with host transfer"
        )

        # Attempt to load custom CUDA extension
        try:
            import histogram_cuda_ext
            self.histogram_cuda_ext = histogram_cuda_ext
        except ImportError:
            try:
                from torch.utils.cpp_extension import load
                cuda_dir = os.path.join(os.path.dirname(__file__), '..', 'cuda')
                sources = [
                    os.path.join(cuda_dir, 'histogram_cuda_ext.cpp'),
                    os.path.join(cuda_dir, 'histogram_kernels.cu')
                ]
                if all(os.path.exists(src) for src in sources):
                    self.histogram_cuda_ext = load(
                        name='histogram_cuda_ext',
                        sources=sources,
                        extra_cflags=['-O3', '-std=c++17'],
                        extra_cuda_cflags=['-O3', '-std=c++17'],
                        verbose=False
                    )
            except Exception as exc:
                print(f"[WARN] Failed to load histogram CUDA extension: {exc}")

        if self.histogram_cuda_ext is not None:
            self.register_implementation(
                "cuda_histogram",
                self._cuda_histogram,
                "CUDA Histogram",
                "Custom CUDA histogram kernel"
            )

        if _is_triton_supported():
            self.register_implementation(
                "triton_histogram",
                self._triton_histogram,
                "Triton Histogram",
                "Triton GPU histogram kernel"
            )
        elif _TRITON_AVAILABLE:
            print("[WARN] Triton histogram kernel requires GPU compute capability 7.0 or higher; skipping registration")

        self.set_reference_implementation("pytorch_bincount")

    def _pytorch_bincount(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        num_bins = params.get("num_bins")
        data = inputs[0]
        try:
            result = torch.bincount(data.to(torch.int64), minlength=num_bins)
            if result.device != data.device:
                result = result.to(data.device)
        except RuntimeError:
            result = torch.bincount(data.cpu().to(torch.int64), minlength=num_bins).to(data.device)
        return result.to(torch.int32)

    def _cpu_bincount(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        num_bins = params.get("num_bins")
        result = torch.bincount(inputs[0].cpu().to(torch.int64), minlength=num_bins)
        return result.to(inputs[0].device, dtype=torch.int32)

    def _cuda_histogram(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        if self.histogram_cuda_ext is None:
            return None
        num_bins = params.get("num_bins")
        return self.histogram_cuda_ext.histogram(inputs[0], num_bins)

    def _triton_histogram(self, inputs: List[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        if not _is_triton_supported():
            return None

        num_bins = params.get("num_bins")
        data = inputs[0]
        if data.dtype != torch.int32:
            data = data.to(torch.int32)
        data = data.contiguous()

        histogram = torch.zeros(num_bins, dtype=torch.int32, device=data.device)
        n = data.numel()
        if n == 0:
            return histogram

        block_size = 1024
        grid = ((n + block_size - 1) // block_size,)
        _histogram_triton_kernel[grid](
            data,
            histogram,
            n,
            num_bins,
            BLOCK_SIZE=block_size,
        )
        return histogram
