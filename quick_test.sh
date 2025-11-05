#!/bin/bash
# 快速测试checkpoint脚本
# 使用方法: bash quick_test.sh <实验目录>

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🧪 Checkpoint 快速测试工具                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供实验目录路径"
    echo ""
    echo "使用方法:"
    echo "  bash quick_test.sh outputs/swin_semi_contrastive_labeled20pct_20251105_103330"
    echo ""
    echo "或查找最新实验:"
    echo "  bash quick_test.sh \$(ls -td outputs/*/ | head -1)"
    exit 1
fi

EXP_DIR="$1"

# 检查目录是否存在
if [ ! -d "$EXP_DIR" ]; then
    echo "❌ 错误: 实验目录不存在: $EXP_DIR"
    exit 1
fi

echo "📂 实验目录: $EXP_DIR"
echo ""

# 激活环境
echo "📦 激活conda环境: swin"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swin

# 设置GPU
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "🎮 使用GPU: $GPU_ID"
echo ""

# 显示找到的checkpoint
echo "🔍 查找checkpoint..."
CKPT_COUNT=$(find "$EXP_DIR" -name "*.ckpt" | wc -l)
echo "✅ 找到 $CKPT_COUNT 个checkpoint"
echo ""

# 列出checkpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Checkpoint列表:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
find "$EXP_DIR" -name "*.ckpt" -exec basename {} \; | nl
echo ""

# 询问是否继续
read -p "🤔 是否开始测试所有checkpoint? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 测试已取消"
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏰ 开始时间: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 运行测试
python test_all_checkpoints.py -e "$EXP_DIR"

TEST_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏰ 结束时间: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成！"
    echo ""
    echo "📄 查看报告:"
    echo "  cat $EXP_DIR/test_report.txt"
    echo ""
    echo "📊 JSON报告:"
    echo "  cat $EXP_DIR/test_report.json"
else
    echo "❌ 测试失败，退出码: $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi
