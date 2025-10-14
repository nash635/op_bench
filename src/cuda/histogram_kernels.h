#pragma once

#include <cuda_runtime.h>

extern "C" void solve(const int* input, int* histogram, int N, int num_bins);
