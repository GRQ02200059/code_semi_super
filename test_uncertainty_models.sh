#!/bin/bash
################################################################################
# 测试不确定性估计模型脚本
################################################################################

EXP_DIR="outputs/swin_semi_uncertainty_labeled20pct_20251105_134618"

echo "=========================================="
echo "🧪 测试不确定性估计模型"
echo "=========================================="
echo ""
echo "实验目录: $EXP_DIR"
echo ""

# 检查环境
echo "🔍 检查环境..."
if ! conda env list | grep -q "^swin "; then
    echo "❌ 错误: swin 环境不存在"
    exit 1
fi

# 激活环境
echo "🔄 激活 swin 环境..."
eval "$(conda shell.bash hook)"
conda activate swin

# 检查目录
if [ ! -d "$EXP_DIR" ]; then
    echo "❌ 错误: 实验目录不存在: $EXP_DIR"
    exit 1
fi

# 统计checkpoint
CKPT_COUNT=$(find $EXP_DIR -name "*.ckpt" | wc -l)
echo "✅ 找到 $CKPT_COUNT 个checkpoint文件"
echo ""

# 显示所有checkpoint
echo "📋 Checkpoint列表:"
echo "----------------------------------------"
find $EXP_DIR -name "*.ckpt" | sort | while read ckpt; do
    basename "$ckpt"
done
echo "----------------------------------------"
echo ""

# 运行测试
echo "🎬 开始测试..."
echo ""

python test_all_checkpoints.py $EXP_DIR

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
