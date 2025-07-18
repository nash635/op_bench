#!/bin/bash
"""
清理脚本 - 清除无用的测试文件和缓存
使用方法: ./scripts/cleanup.sh [options]
"""

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 帮助信息
show_help() {
    echo "清理脚本 - 清除无用的测试文件和缓存"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -a, --all          全面清理（包括测试结果）"
    echo "  -c, --cache        只清理缓存文件"
    echo "  -t, --test-results 只清理测试结果"
    echo "  -e, --empty        清理空文件"
    echo "  -h, --help         显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 -c              # 只清理缓存"
    echo "  $0 -a              # 全面清理"
    echo "  $0 --test-results  # 只清理测试结果"
}

# 清理Python缓存
cleanup_python_cache() {
    log_info "清理Python缓存文件..."
    
    # 清理 __pycache__ 目录
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # 清理 .pyc 文件
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    
    # 清理 .pyo 文件
    find . -name "*.pyo" -type f -delete 2>/dev/null || true
    
    log_success "Python缓存清理完成"
}

# 清理构建缓存
cleanup_build_cache() {
    log_info "清理构建缓存..."
    
    # 清理CMake缓存
    if [ -d "build" ]; then
        rm -rf build/
        log_success "已删除 build/ 目录"
    fi
    
    # 清理PyTorch JIT缓存
    find . -name ".torch_compile" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "构建缓存清理完成"
}

# 清理测试结果（保留最新的5个）
cleanup_test_results() {
    log_info "清理旧的测试结果文件..."
    
    if [ -d "test_results" ]; then
        cd test_results
        
        # 计算文件数量
        total_files=$(ls -1 *.json *.md *.png 2>/dev/null | wc -l)
        
        if [ "$total_files" -gt 5 ]; then
            # 保留最新的5个文件，删除其余的
            ls -t *.json *.md *.png 2>/dev/null | tail -n +6 | xargs rm -f
            remaining_files=$(ls -1 *.json *.md *.png 2>/dev/null | wc -l)
            log_success "删除了 $((total_files - remaining_files)) 个旧测试结果文件，保留最新的 $remaining_files 个"
        else
            log_info "测试结果文件数量合理，无需清理"
        fi
        
        cd ..
    else
        log_warning "test_results 目录不存在"
    fi
}

# 清理空文件
cleanup_empty_files() {
    log_info "清理空文件..."
    
    # 查找空的Python文件
    empty_py_files=$(find . -name "*.py" -type f -size 0 2>/dev/null)
    if [ -n "$empty_py_files" ]; then
        echo "$empty_py_files" | xargs rm -f
        log_success "删除了空的Python文件"
    else
        log_info "未找到空的Python文件"
    fi
    
    # 查找空的文本文件
    empty_txt_files=$(find . -name "*.txt" -type f -size 0 2>/dev/null)
    if [ -n "$empty_txt_files" ]; then
        echo "$empty_txt_files" | xargs rm -f
        log_success "删除了空的文本文件"
    else
        log_info "未找到空的文本文件"
    fi
}

# 清理临时文件
cleanup_temp_files() {
    log_info "清理临时文件..."
    
    # 清理编辑器临时文件
    find . -name "*~" -type f -delete 2>/dev/null || true
    find . -name "*.swp" -type f -delete 2>/dev/null || true
    find . -name "*.swo" -type f -delete 2>/dev/null || true
    find . -name ".DS_Store" -type f -delete 2>/dev/null || true
    
    # 清理matplotlib临时缓存
    find /tmp -name "matplotlib_cache_*" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "临时文件清理完成"
}

# 检查工作目录
check_working_directory() {
    if [ ! -f "README.md" ] || [ ! -d "src" ]; then
        log_error "请在op_bench项目根目录下运行此脚本"
        exit 1
    fi
}

# 主函数
main() {
    check_working_directory
    
    log_info "开始清理op_bench项目..."
    echo
    
    # 解析参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--cache)
            cleanup_python_cache
            cleanup_build_cache
            ;;
        -t|--test-results)
            cleanup_test_results
            ;;
        -e|--empty)
            cleanup_empty_files
            ;;
        -a|--all)
            cleanup_python_cache
            cleanup_build_cache
            cleanup_test_results
            cleanup_empty_files
            cleanup_temp_files
            ;;
        "")
            # 默认清理：缓存 + 旧测试结果
            cleanup_python_cache
            cleanup_test_results
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
    
    echo
    log_success "清理完成！"
}

# 运行主函数
main "$@"
