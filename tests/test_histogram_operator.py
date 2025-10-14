#!/usr/bin/env python3
"""Quick integration test for HistogramOperator"""

import sys
import os
import torch

# Ensure src directory is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_histogram_operator():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable; skipping histogram operator test")
        return True

    from operators.histogram_operator import HistogramOperator

    op = HistogramOperator()
    test_case = op.get_test_cases()[0]
    inputs = op.generate_inputs(test_case)
    reference = op.get_reference_result(inputs, test_case)

    successes = True

    # Validate PyTorch baseline implementation
    pytorch_result = op.run_implementation('pytorch_bincount', inputs, test_case.additional_params)
    if not torch.equal(reference, pytorch_result):
        print("[FAIL] PyTorch baseline does not match reference")
        successes = False

    if 'cuda_histogram' in op.get_available_implementations():
        cuda_result = op.run_implementation('cuda_histogram', inputs, test_case.additional_params)
        aligned_cuda = cuda_result.to(device=reference.device, dtype=torch.int32)
        if not torch.equal(reference, aligned_cuda):
            diff = (aligned_cuda.to(torch.int64) - reference.to(torch.int64)).abs().max().item()
            print(f"[FAIL] CUDA histogram mismatch: max diff {diff}")
            successes = False
        else:
            print("[PASS] CUDA histogram matches reference")
    else:
        print("[WARN] CUDA histogram implementation not registered")

    if 'triton_histogram' in op.get_available_implementations():
        triton_result = op.run_implementation('triton_histogram', inputs, test_case.additional_params)
        aligned_triton = triton_result.to(device=reference.device, dtype=torch.int32)
        if not torch.equal(reference, aligned_triton):
            diff = (aligned_triton.to(torch.int64) - reference.to(torch.int64)).abs().max().item()
            print(f"[FAIL] Triton histogram mismatch: max diff {diff}")
            successes = False
        else:
            print("[PASS] Triton histogram matches reference")
    else:
        print("[WARN] Triton histogram implementation not registered")

    return successes


if __name__ == "__main__":
    success = test_histogram_operator()
    sys.exit(0 if success else 1)
