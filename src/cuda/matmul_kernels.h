#pragma once

#include <torch/extension.h>

// Base CUDA matmul function declarations
torch::Tensor matmul_basic_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_shared_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_static_shared_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_template_cuda(torch::Tensor A, torch::Tensor B, int tile_width);

// CUTLASS-style optimized matmul declaration
torch::Tensor cutlass_style_matmul_cuda(torch::Tensor A, torch::Tensor B);
