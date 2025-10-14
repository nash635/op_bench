#include "histogram_kernels.h"

#include <algorithm>
#include <cstdio>

namespace {

// Optimization variant 1: use privatization to reduce shared memory contention
__global__ void histogram_kernel_optimized(const int* __restrict__ input,
                                           int* __restrict__ histogram,
                                           int N,
                                           int num_bins) {
    extern __shared__ int shared_hist[];

    // Zero-initialize shared memory
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_hist[idx] = 0;
    }
    __syncthreads();

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Strategy 1: privatization for small num_bins
    if (num_bins <= 256) {
        // Each thread maintains a private histogram
        int private_hist[256];
        #pragma unroll
        for (int i = 0; i < num_bins; i++) {
            private_hist[i] = 0;
        }

    // Accumulate within the private histogram
        for (int i = global_thread_id; i < N; i += stride) {
            int bin = input[i];
            if (bin >= 0 && bin < num_bins) {
                private_hist[bin]++;
            }
        }

    // Merge private histograms into shared memory
        for (int i = 0; i < num_bins; i++) {
            if (private_hist[i] > 0) {
                atomicAdd(&shared_hist[i], private_hist[i]);
            }
        }
    } else {
    // Strategy 2: use shared-memory atomics for large num_bins
        for (int i = global_thread_id; i < N; i += stride) {
            int bin = input[i];
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&shared_hist[bin], 1);
            }
        }
    }

    __syncthreads();

    // Flush aggregated counts to global memory
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        int count = shared_hist[idx];
        if (count != 0) {
            atomicAdd(&histogram[idx], count);
        }
    }
}

// Optimization variant 2: tuned for small bin counts (leveraging warp-level behavior)
__global__ void histogram_kernel_small_bins(const int* __restrict__ input,
                                            int* __restrict__ histogram,
                                            int N,
                                            int num_bins) {
    extern __shared__ int shared_hist[];

    // Initialize the shared histogram with all threads
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_hist[idx] = 0;
    }
    __syncthreads();

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Use a register array when the size is known at compile time
    int private_hist[64] = {0}; // Assumes num_bins <= 64

    // Accumulate in thread-local storage
    for (int i = global_thread_id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            private_hist[bin]++;
        }
    }

    // Merge into shared memory
    #pragma unroll
    for (int i = 0; i < 64 && i < num_bins; i++) {
        if (private_hist[i] > 0) {
            atomicAdd(&shared_hist[i], private_hist[i]);
        }
    }

    __syncthreads();

    // Write results back to global memory
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        if (shared_hist[idx] != 0) {
            atomicAdd(&histogram[idx], shared_hist[idx]);
        }
    }
}

// Optimization variant 3: segmented processing for large bin counts
__global__ void histogram_kernel_large_bins(const int* __restrict__ input,
                                            int* __restrict__ histogram,
                                            int N,
                                            int num_bins) {
    extern __shared__ int shared_hist[];

    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_hist[idx] = 0;
    }
    __syncthreads();

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Use shared memory directly to reduce register pressure
    for (int i = global_thread_id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();

    // Flush in batches using vectorized loads
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        int count = shared_hist[idx];
        if (count != 0) {
            atomicAdd(&histogram[idx], count);
        }
    }
}

}  // namespace

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (histogram == nullptr || input == nullptr || num_bins <= 0) {
        return;
    }

    cudaError_t err = cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[histogram] cudaMemset failed: %s\n", cudaGetErrorString(err));
        return;
    }

    if (N <= 0) {
        return;
    }

    // Dynamically choose the best kernel configuration
    int threads_per_block;
    int blocks;
    size_t shared_mem_bytes = static_cast<size_t>(num_bins) * sizeof(int);

    // Query device shared-memory limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (shared_mem_bytes > prop.sharedMemPerBlock) {
        std::fprintf(stderr, "[histogram] num_bins too large for shared memory\n");
        return;
    }

    if (num_bins <= 64) {
        // Small bin counts: more threads with privatization
        threads_per_block = 256;
        blocks = std::min((N + threads_per_block - 1) / threads_per_block, 
                         std::min(prop.multiProcessorCount * 4, 65535));
        histogram_kernel_small_bins<<<blocks, threads_per_block, shared_mem_bytes>>>(
            input, histogram, N, num_bins);
    } else if (num_bins <= 256) {
        // Medium bin counts: optimized privatization strategy
        threads_per_block = 256;
        blocks = std::min((N + threads_per_block - 1) / threads_per_block,
                         std::min(prop.multiProcessorCount * 2, 65535));
        histogram_kernel_optimized<<<blocks, threads_per_block, shared_mem_bytes>>>(
            input, histogram, N, num_bins);
    } else {
        // Large bin counts: limit blocks to reduce global atomic contention
        threads_per_block = 256;
        blocks = std::min((N + threads_per_block - 1) / threads_per_block,
                         std::min(prop.multiProcessorCount, 1024));
        histogram_kernel_large_bins<<<blocks, threads_per_block, shared_mem_bytes>>>(
            input, histogram, N, num_bins);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[histogram] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[histogram] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }
}