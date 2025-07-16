#!/bin/bash
#
# 运行Nsight Profiling的包装脚本
# 自动设置正确的库路径以确保CUDA扩展能够正常加载
#

# 获取PyTorch库路径
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))")

# 设置环境变量
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "设置库路径: $TORCH_LIB_PATH"
echo "设置Python路径: $(pwd)"
echo "运行profiling脚本..."
echo "----------------------------------------"

# 运行profiling脚本，传递所有参数
python tools/profile_nsight.py "$@"
