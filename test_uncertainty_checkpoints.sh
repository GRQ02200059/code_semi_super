#!/bin/bash
################################################################################
# 不确定性估计模型测试脚本
################################################################################

echo "=========================================="
echo "🧪 测试不确定性估计模型"
echo "=========================================="
echo ""

# 1. 激活环境
echo "🔄 激活 swin 环境..."
eval "$(conda shell.bash hook)"
conda activate swin
echo "✅ 环境已激活"
echo ""

# 2. 设置GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "⚙️  设置 CUDA_VISIBLE_DEVICES=0"
else
    echo "⚙️  使用现有 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
echo ""

# 3. 实验目录
EXP_DIR="outputs/swin_semi_uncertainty_labeled20pct_20251105_134618"

echo "=========================================="
echo "📋 测试配置"
echo "=========================================="
echo "实验目录: $EXP_DIR"
echo "检查点数量: $(find $EXP_DIR -name "*.ckpt" | wc -l)"
echo ""

echo "发现的检查点:"
find $EXP_DIR -name "*.ckpt" | sort | while read ckpt; do
    echo "  - $(basename $ckpt)"
done
echo ""

# 4. 运行测试
echo "=========================================="
echo "🚀 开始测试所有检查点"
echo "=========================================="
echo ""

python test_all_checkpoints.py $EXP_DIR

# 5. 完成
echo ""
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "🎉 测试完成！"
    echo "=========================================="
    echo ""
    echo "📁 结果保存在: ${EXP_DIR}/test_results/"
    echo ""
    echo "查看结果:"
    echo "  - 汇总报告: ${EXP_DIR}/test_results/summary.txt"
    echo "  - 详细CSV: ${EXP_DIR}/test_results/all_results.csv"
    echo "  - 最佳模型: ${EXP_DIR}/test_results/best_checkpoint.txt"
    echo ""
else
    echo "=========================================="
    echo "❌ 测试失败"
    echo "=========================================="
    exit 1
fi
