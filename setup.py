from setuptools import setup, find_packages
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

# 读取requirements.txt中的依赖
def read_requirements():
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
    
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # 处理可选依赖的注释
            if not line.startswith('# '):
                requirements.append(line)
    return requirements

setup(
    name='op_bench',
    version='0.1.0',
    description='Operator Benchmark Framework with FP8 Linear Support',
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        'fp8': [
            'transformer-engine>=1.0.0',
        ]
    },
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
