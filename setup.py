from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 源文件列表
sources = [
    'src/cuda/matmul_cuda_ext.cpp',
    'src/cuda/matmul_kernels.cu',
]

# 编译参数
extra_compile_args_cxx = ['-std=c++17', '-O3']
extra_compile_args_nvcc = ['-std=c++17', '-O3', '--expt-relaxed-constexpr']

# 包含路径
include_dirs = []

setup(
    name='optimal_matmul_cuda',
    ext_modules=[
        CUDAExtension(
            name='matmul_cuda_ext',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': extra_compile_args_cxx,
                'nvcc': extra_compile_args_nvcc
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    python_requires=">=3.6",
)
