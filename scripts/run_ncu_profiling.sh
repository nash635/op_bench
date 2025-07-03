#!/bin/bash
# CUDA MatMul Nsight Compute Profiling 便捷脚本
# 
# 使用方法:
#   ./run_ncu_profiling.sh                    # 默认配置
#   ./run_ncu_profiling.sh 1024               # 指定矩阵大小
#   ./run_ncu_profiling.sh 512 1024           # 多个矩阵大小

echo "CUDA MatMul Nsight Compute Profiling"
echo "====================================="

# 检查 CUDA 扩展是否已构建
if [ ! -f matmul_cuda_ext.*.so ]; then
    echo "错误: CUDA 扩展未找到，请先运行 ./build.sh"
    exit 1
fi

# 检查 ncu 是否可用
if ! command -v ncu &> /dev/null; then
    echo "错误: ncu 命令未找到，请确保 Nsight Compute 已安装"
    exit 1
fi

# 解析参数
SIZES=${@:-"1024"}  # 默认矩阵大小
OUTPUT_DIR="ncu_profiles_$(date +%Y%m%d_%H%M%S)"
KERNELS="pytorch_builtin cuda_basic cuda_shared cuda_template_32"

echo "配置:"
echo "  矩阵大小: $SIZES"
echo "  输出目录: $OUTPUT_DIR"
echo "  分析 kernels: $KERNELS"
echo ""

# 首先尝试 NCU profiling
echo "尝试运行 Nsight Compute profiling..."
sudo python ncu_profiler.py \
    --sizes $SIZES \
    --output $OUTPUT_DIR \
    --kernels $KERNELS \
    --metrics default

NCU_EXIT_CODE=$?

if [ $NCU_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "NCU Profiling 完成！"
    echo ""
    echo "下一步:"
    echo "  1. 运行对比分析: cd $OUTPUT_DIR && ./compare_ncu_profiles.sh"
    echo "  2. GUI 详细分析: ncu-ui $OUTPUT_DIR/*.ncu-rep"
    echo "  3. 查看指标数据: cat $OUTPUT_DIR/*_metrics.csv"
    echo ""
else
    echo ""
    echo "NCU Profiling 失败，可能是权限问题。"
    echo "切换到 PyTorch Profiler 替代方案..."
    echo ""
    
    # 使用替代方案
    ALT_OUTPUT_DIR="pytorch_profiler_$(date +%Y%m%d_%H%M%S)"
    python profiling_alternative.py --sizes $SIZES --output $ALT_OUTPUT_DIR --force-pytorch-profiler
    
    ALT_EXIT_CODE=$?
    
    if [ $ALT_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "PyTorch Profiler 分析完成！"
        echo ""
        echo "下一步:"
        echo "  1. 查看分析报告: cat $ALT_OUTPUT_DIR/pytorch_profiler_report.md"
        echo "  2. Chrome 可视化: 打开 Chrome 浏览器的 chrome://tracing/ 并加载 .json 文件"
        echo "  3. 解决 NCU 权限: 参考报告中的权限设置指导"
        echo ""
    else
        echo "所有分析方法都失败了，请检查环境配置"
        exit 1
    fi
fi
