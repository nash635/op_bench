#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 辅助函数
__device__ __host__ inline int cdiv(int a, int b) { 
    return (a + b - 1) / b; 
}


__global__ void matmul_basic_kernel(float* m, float* n, float* out, int h, int w, int k) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;

    if (r>=h || c>=w) return;
    float o = 0;
    for (int i = 0; i<k; ++i) o += m[r*k+i] * n[i*w+c];
    out[r*w+c] = o;
}



__global__ void matmul_shared_kernel(float *m, float *n, float *out, int h, int w, int k, int tw) {
    int tc=threadIdx.x, tr=threadIdx.y;
    int r=blockIdx.y*blockDim.y+tr, c=blockIdx.x*blockDim.x+tc;

    extern __shared__ float ms[];
    float *ns = &ms[tw*tw];

    float p = 0.0f;
    for (int ph = 0; ph < cdiv(k,tw); ++ph) {
        int idx = ph*tw;
        ms[tr*tw + tc] = r<h && idx+tc<k ? m[ tc+idx + r*k ] : 0.0f;
        ns[tr*tw + tc] = c<w && idx+tr<k ? n[(tr+idx)*w + c] : 0.0f;
        __syncthreads();
        for (int i=0; i<tw; ++i) p += ms[tr*tw + i] * ns[tw*i + tc];
        __syncthreads();
    }
    if (r<h && c<w) out[r*w + c] = p;
}



constexpr int tw = 16;

__global__ void matmul_static_shared_kernel(float *m, float *n, float *out, int h, int w, int k) {
    __shared__ float ms[tw][tw], ns[tw][tw];
    int tc=threadIdx.x, tr=threadIdx.y;
    int r=blockIdx.y*blockDim.y+tr, c=blockIdx.x*blockDim.x+tc;

    float p=0.0f;
    for (int ph=0; ph < cdiv(k,tw); ++ph) {
        int idx = ph*tw;
        ms[tr][tc] = r<h && idx+tc<k ? m[ tc+idx + r*k ] : 0.0f;
        ns[tr][tc] = c<w && idx+tr<k ? n[(tr+idx)*w + c] : 0.0f;
        __syncthreads();
        for (int i=0; i<tw; ++i) p += ms[tr][i] * ns[i][tc];
        __syncthreads();
    }
    if (r<h && c<w) out[r*w + c] = p;
}



template<int TW>
__global__ void matmul_template_kernel(float *m, float *n, float *out, int h, int w, int k) {
    int tc=threadIdx.x, tr=threadIdx.y;
    int r=blockIdx.y*blockDim.y+tr, c=blockIdx.x*blockDim.x+tc;
    
    extern __shared__ float ms[];
    float *ns = &ms[TW*TW];

    float p = 0.0f;
    for (int ph = 0; ph < cdiv(k,TW); ++ph) {
        int idx = ph*TW;
        ms[tr*TW + tc] = r<h && idx+tc<k ? m[ tc+idx + r*k ] : 0.0f;
        ns[tr*TW + tc] = c<w && idx+tr<k ? n[(tr+idx)*w + c] : 0.0f;
        __syncthreads();
        for (int i=0; i<TW; ++i) p += ms[tr*TW + i] * ns[TW*i + tc];
        __syncthreads();
    }
    if (r<h && c<w) out[r*w + c] = p;
}


// CUTLASS-style 优化 GEMM kernel
template<int TILE_WIDTH>
__global__ void cutlass_style_gemm_kernel(
    float* __restrict__ A, 
    float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K) {
    
    // 使用更大的 shared memory
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float result = 0.0f;
    
    // 优化的循环展开和预取
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // 协作加载到 shared memory
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_WIDTH + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 展开计算循环以提高 ILP (Instruction Level Parallelism)
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            result += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = result;
    }
}

// PyTorch接口函数

torch::Tensor matmul_basic_cuda(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0), w = n.size(1), k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16,16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_basic_kernel<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    return output;
}

torch::Tensor matmul_shared_cuda(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0), w = n.size(1), k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    int TW = 16;
    size_t size = TW*TW*2 * sizeof(float);
    dim3 tpb(TW,TW);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_shared_kernel<<<blocks,tpb,size>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k, TW);
    return output;
}

torch::Tensor matmul_static_shared_cuda(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0), w = n.size(1), k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(tw,tw);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_static_shared_kernel<<<blocks,tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    return output;
}

torch::Tensor matmul_template_cuda(torch::Tensor m, torch::Tensor n, int tile_width) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0), w = n.size(1), k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    size_t size = tile_width*tile_width*2 * sizeof(float);
    dim3 tpb(tile_width,tile_width);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    
    switch(tile_width) {
        case 8: matmul_template_kernel<8><<<blocks,tpb,size>>>(
            m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k); break;
        case 16: matmul_template_kernel<16><<<blocks,tpb,size>>>(
            m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k); break;
        case 32: matmul_template_kernel<32><<<blocks,tpb,size>>>(
            m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k); break;
        default: TORCH_CHECK(false, "Unsupported tile width");
    }
    return output;
}

torch::Tensor cutlass_style_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions must match");
    
    auto output = torch::zeros({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    
    // 使用 32x32 的 tile size 以优化内存访问
    constexpr int TILE_WIDTH = 32;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    cutlass_style_gemm_kernel<TILE_WIDTH><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(),
        M, N, K);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}
