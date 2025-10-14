#include <torch/extension.h>
#include <limits>

#include "histogram_kernels.h"

namespace {

torch::Tensor histogram_cuda(torch::Tensor input, int64_t num_bins) {
    TORCH_CHECK(input.is_cuda(), "input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kInt32, "input tensor must have dtype torch.int32");
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
    TORCH_CHECK(num_bins > 0, "num_bins must be positive");
    TORCH_CHECK(num_bins <= 1024, "num_bins must not exceed 1024");
    TORCH_CHECK(input.numel() <= std::numeric_limits<int>::max(), "input tensor is too large");

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(input.device());

    torch::Tensor histogram = torch::empty({num_bins}, options);
    solve(input.data_ptr<int>(), histogram.data_ptr<int>(), static_cast<int>(input.numel()), static_cast<int>(num_bins));
    return histogram;
}

torch::Tensor histogram_cuda_from_view(torch::Tensor input, int64_t num_bins) {
    auto contiguous = input.contiguous();
    return histogram_cuda(contiguous, num_bins);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("histogram", &histogram_cuda_from_view, "CUDA histogram computation (int32 input)");
}
