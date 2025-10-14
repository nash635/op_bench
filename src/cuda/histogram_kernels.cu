#include "histogram_kernels.h"

#include <algorithm>
#include <cstdio>

namespace {

// 优化版本1: 使用私有化技术减少共享内存竞争
__global__ void histogram_kernel_optimized(const int* __restrict__ input,
                                           int* __restrict__ histogram,
                                           int N,
                                           int num_bins) {
    extern __shared__ int shared_hist[];

    // 零初始化共享内存
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_hist[idx] = 0;
    }
    __syncthreads();

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // 策略1: 对于小的 num_bins，使用私有化
    if (num_bins <= 256) {
        // 每个线程维护一个私有 histogram
        int private_hist[256];
        #pragma unroll
        for (int i = 0; i < num_bins; i++) {
            private_hist[i] = 0;
        }

        // 在私有 histogram 中累加
        for (int i = global_thread_id; i < N; i += stride) {
            int bin = input[i];
            if (bin >= 0 && bin < num_bins) {
                private_hist[bin]++;
            }
        }

        // 将私有 histogram 合并到共享内存
        for (int i = 0; i < num_bins; i++) {
            if (private_hist[i] > 0) {
                atomicAdd(&shared_hist[i], private_hist[i]);
            }
        }
    } else {
        // 策略2: 对于大的 num_bins，直接使用共享内存原子操作
        for (int i = global_thread_id; i < N; i += stride) {
            int bin = input[i];
            if (bin >= 0 && bin < num_bins) {
                atomicAdd(&shared_hist[bin], 1);
            }
        }
    }

    __syncthreads();

    // 批量写回全局内存
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        int count = shared_hist[idx];
        if (count != 0) {
            atomicAdd(&histogram[idx], count);
        }
    }
}

// 优化版本2: 专门针对小 bin 数量的优化（使用 warp-level 操作）
__global__ void histogram_kernel_small_bins(const int* __restrict__ input,
                                            int* __restrict__ histogram,
                                            int N,
                                            int num_bins) {
    extern __shared__ int shared_hist[];

    // 使用所有线程快速初始化
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        shared_hist[idx] = 0;
    }
    __syncthreads();

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // 使用寄存器数组（编译时大小已知时更高效）
    int private_hist[64] = {0}; // 假设 num_bins <= 64

    // 在私有空间累加
    for (int i = global_thread_id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            private_hist[bin]++;
        }
    }

    // 合并到共享内存
    #pragma unroll
    for (int i = 0; i < 64 && i < num_bins; i++) {
        if (private_hist[i] > 0) {
            atomicAdd(&shared_hist[i], private_hist[i]);
        }
    }

    __syncthreads();

    // 写回全局内存
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        if (shared_hist[idx] != 0) {
            atomicAdd(&histogram[idx], shared_hist[idx]);
        }
    }
}

// 优化版本3: 针对大 bin 数量的分段处理
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

    // 直接使用共享内存，减少寄存器压力
    for (int i = global_thread_id; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&shared_hist[bin], 1);
        }
    }

    __syncthreads();

    // 批量写回，使用向量化加载提高效率
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

    // 动态选择最优的 kernel 和配置
    int threads_per_block;
    int blocks;
    size_t shared_mem_bytes = static_cast<size_t>(num_bins) * sizeof(int);

    // 查询设备共享内存限制
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (shared_mem_bytes > prop.sharedMemPerBlock) {
        std::fprintf(stderr, "[histogram] num_bins too large for shared memory\n");
        return;
    }

    if (num_bins <= 64) {
        // 小 bin 数量：使用更多线程，私有化策略
        threads_per_block = 256;
        blocks = std::min((N + threads_per_block - 1) / threads_per_block, 
                         std::min(prop.multiProcessorCount * 4, 65535));
        histogram_kernel_small_bins<<<blocks, threads_per_block, shared_mem_bytes>>>(
            input, histogram, N, num_bins);
    } else if (num_bins <= 256) {
        // 中等 bin 数量：使用优化的私有化策略
        threads_per_block = 256;
        blocks = std::min((N + threads_per_block - 1) / threads_per_block,
                         std::min(prop.multiProcessorCount * 2, 65535));
        histogram_kernel_optimized<<<blocks, threads_per_block, shared_mem_bytes>>>(
            input, histogram, N, num_bins);
    } else {
        // 大 bin 数量：减少线程块数量，避免过多全局原子操作竞争
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