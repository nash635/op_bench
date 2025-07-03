#!/bin/bash

# 强制使用支持C++17的编译器
export CC=g++
export CXX=g++

# 定义Python解释器路径
PYTHON_EXECUTABLE="/home/jian.sha/miniconda3/envs/py310/bin/python"

echo "=========================================="
echo "Optimal CUDA MatMul Extension Build Script"
echo "=========================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: CUDA编译器(nvcc)未找到"
    exit 1
fi

# 检查PyTorch
$PYTHON_EXECUTABLE -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "错误: PyTorch未安装"
    exit 1
}

# 检查CUDA可用性
$PYTHON_EXECUTABLE -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'" || {
    echo "错误: CUDA在PyTorch中不可用"
    exit 1
}

echo "✓ 环境检查通过"

# 清理之前的构建
echo "清理之前的构建文件..."
rm -rf build/ dist/ *.egg-info optimal_matmul_cuda.egg-info matmul_cuda_ext.cpython-310-x86_64-linux-gnu.so

# 运行构建 (原地构建)
echo "运行构建..."
$PYTHON_EXECUTABLE setup.py build_ext --inplace

if [ $? -ne 0 ]; then
    echo "✗ 构建失败"
    exit 1
fi

echo "✓ 构建成功"

# 运行测试
echo "运行正确性测试..."
$PYTHON_EXECUTABLE test_extension.py || {
    echo "测试失败!"
    exit 1
}

echo "运行快速性能测试..."
$PYTHON_EXECUTABLE -c "
import torch
import matmul_cuda_ext # 直接导入
import time

try:
    print('加载扩展...')
    size = 1024
    A = torch.randn(size, size, device='cuda').contiguous()
    B = torch.randn(size, size, device='cuda').contiguous()

    # 预热
    for _ in range(3):
        _ = torch.mm(A, B)
        _ = matmul_cuda_ext.matmul_static_shared(A, B)
        torch.cuda.synchronize()

    # 测试
    start = time.time()
    for _ in range(10):
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 10

    start = time.time()
    for _ in range(10):
        _ = matmul_cuda_ext.matmul_static_shared(A, B)
        torch.cuda.synchronize()
    our_time = (time.time() - start) / 10

    flops = 2 * size**3
    print(f'快速性能测试 ({size}x{size}):')
    print(f'  PyTorch: {pytorch_time*1000:.2f} ms ({flops/(pytorch_time*1e9):.1f} GFLOPS)')
    print(f'  我们的:  {our_time*1000:.2f} ms ({flops/(our_time*1e9):.1f} GFLOPS)')
    print(f'  加速比: {pytorch_time/our_time:.2f}x')

except Exception as e:
    print(f'快速性能测试失败: {e}')
"

echo ""
echo "=========================================="
echo "构建和测试完成!"
echo "要运行完整基准测试，执行:"
echo "  /home/jian.sha/miniconda3/envs/py310/bin/python benchmark_matmul.py -o results/"
echo "=========================================="
