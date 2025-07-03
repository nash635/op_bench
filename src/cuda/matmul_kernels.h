#pragma once

#include <torch/extension.h>

// 基础 CUDA matmul 函数声明
torch::Tensor matmul_basic_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_shared_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_static_shared_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_template_cuda(torch::Tensor A, torch::Tensor B, int tile_width);

// CUTLASS-style 优化 matmul 函数声明
torch::Tensor cutlass_style_matmul_cuda(torch::Tensor A, torch::Tensor B);
