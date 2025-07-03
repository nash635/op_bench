#!/bin/bash
# CUDA MatMul Nsight Systems Profiling 便捷脚本
# 
# 使用方法:
#   ./run_profiling.sh                    # 默认配置
#   ./run_profiling.sh 1024 2048         # 指定矩阵大小
#   ./run_profiling.sh 512 1024 2048     # 多个矩阵大小

echo "CUDA MatMul Nsight Systems Profiling"
echo "======================================="

# 检查 CUDA 扩展是否已构建
if [ ! -f matmul_cuda_ext.*.so ]; then
    echo "错误: CUDA 扩展未找到，请先运行 ./build.sh"
    exit 1
fi

# 检查 nsys 是否可用
if ! command -v nsys &> /dev/null; then
    echo "错误: nsys 命令未找到，请确保 Nsight Systems 已安装"
    exit 1
fi

# 解析参数
SIZES=${@:-"512 1024"}  # 默认矩阵大小
OUTPUT_DIR="nsight_profiles_$(date +%Y%m%d_%H%M%S)"
ITERATIONS=100

echo "配置:"
echo "  矩阵大小: $SIZES"
echo "  输出目录: $OUTPUT_DIR"
echo "  迭代次数: $ITERATIONS"
echo ""

# 运行 profiling
python profile_nsight.py \
    --sizes $SIZES \
    --iterations $ITERATIONS \
    --output $OUTPUT_DIR \
    --kernels pytorch_builtin cuda_basic cuda_shared cuda_static_shared cuda_template_32

if [ $? -eq 0 ]; then
    echo ""
    echo "Profiling 完成！"
    echo ""
    echo "下一步:"
    echo "  1. 查看总结报告: cat $OUTPUT_DIR/profiling_summary.md"
    echo "  2. 运行分析脚本: cd $OUTPUT_DIR && ./analyze_profiles.sh"
    echo "  3. GUI 分析: nsight-sys $OUTPUT_DIR/*.nsys-rep"
    echo ""
else
    echo "Profiling 失败"
    exit 1
fi
