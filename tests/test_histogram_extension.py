#!/usr/bin/env python3
"""Tests for the histogram CUDA extension"""

import sys
import os
import torch

# Add src directory to path for fallback loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _load_extension():
    try:
        import histogram_cuda_ext  # type: ignore
        return histogram_cuda_ext
    except ImportError:
        from torch.utils.cpp_extension import load
        cuda_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda')
        sources = [
            os.path.join(cuda_dir, 'histogram_cuda_ext.cpp'),
            os.path.join(cuda_dir, 'histogram_kernels.cu'),
        ]
        return load(
            name='histogram_cuda_ext',
            sources=sources,
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-std=c++17'],
            verbose=False,
        )


def _reference_histogram(input_tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
    counts = torch.bincount(input_tensor.to(torch.int64), minlength=num_bins)
    return counts.to(dtype=torch.int32)


def _run_case(ext_module, data: torch.Tensor, num_bins: int) -> bool:
    if not data.is_cuda:
        data = data.cuda()
    result = ext_module.histogram(data.contiguous(), num_bins)
    reference = _reference_histogram(data, num_bins).to(device=data.device)
    if result.dtype != torch.int32:
        print(f"[FAIL] Unexpected dtype {result.dtype}")
        return False
    if not torch.equal(result.cpu(), reference.cpu()):
        max_diff = (result.to(torch.int32) - reference.to(torch.int32)).abs().max().item()
        print(f"[FAIL] Histogram mismatch (max diff {max_diff})")
        return False
    return True


def test_histogram_extension():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available; skipping histogram extension tests")
        return True

    print("=" * 60)
    print("Histogram CUDA Extension Tests")
    print("=" * 60)

    ext_module = _load_extension()
    torch.manual_seed(42)

    cases = [
        (torch.tensor([0, 1, 2, 1, 0], dtype=torch.int32, device='cuda'), 3, "small_fixed"),
        (torch.randint(0, 16, (1 << 12,), dtype=torch.int32, device='cuda'), 16, "random_medium"),
        (torch.randint(0, 1024, (1 << 20,), dtype=torch.int32, device='cuda'), 1024, "random_large"),
    ]

    all_passed = True
    for data, bins, name in cases:
        print(f"[INFO] Running case '{name}' with {data.numel()} elements and {bins} bins")
        if _run_case(ext_module, data, bins):
            print("  [PASS] Match with reference")
        else:
            print("  [FAIL] Case failed")
            all_passed = False

    print("[RESULT] Histogram extension tests " + ("passed" if all_passed else "failed"))
    return all_passed


if __name__ == "__main__":
    success = test_histogram_extension()
    sys.exit(0 if success else 1)
