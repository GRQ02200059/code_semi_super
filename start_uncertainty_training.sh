#!/bin/bash
################################################################################
# Swin Transformer + 对比学习 + 不确定性估计 训练启动脚本
################################################################################

echo "=========================================="
echo "🚀 Swin Transformer + 不确定性估计 训练启动"
echo "=========================================="
echo ""

# 1. 检查环境
echo "🔍 检查环境..."
if ! conda env list | grep -q "^swin "; then
    echo "❌ 错误: swin 环境不存在"
    echo "   请先创建环境: conda create -n swin python=3.10"
    exit 1
fi
echo "✅ swin 环境存在"
echo ""

# 2. 激活环境
echo "🔄 激活 swin 环境..."
eval "$(conda shell.bash hook)"
conda activate swin
echo "✅ 环境已激活: $(which python)"
echo ""

# 3. 检查数据
echo "🔍 检查数据集..."
if [ ! -d "data/ems" ]; then
    echo "❌ 错误: data/ems 目录不存在"
    echo "   请确保数据集已正确准备"
    exit 1
fi
echo "✅ 数据集存在"
echo ""

# 4. 检查配置文件
echo "🔍 检查配置文件..."
if [ ! -f "configs/semi_supervised/ems_swin_semi_uncertainty.py" ]; then
    echo "❌ 错误: 配置文件不存在"
    exit 1
fi
echo "✅ 配置文件存在"
echo ""

# 5. 运行测试（可选）
echo "🧪 运行集成测试（按 Ctrl+C 跳过）..."
read -t 5 -p "是否运行测试? (y/N) " run_test || run_test="N"
echo ""
if [[ "$run_test" =~ ^[Yy]$ ]]; then
    python test_uncertainty_setup.py
    if [ $? -ne 0 ]; then
        echo "❌ 测试失败，是否继续训练? (y/N)"
        read continue_train
        if [[ ! "$continue_train" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
fi

# 6. 设置GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "⚙️  设置 CUDA_VISIBLE_DEVICES=0"
else
    echo "⚙️  使用现有 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
echo ""

# 7. 打印配置摘要
echo "=========================================="
echo "📋 训练配置摘要"
echo "=========================================="
echo "模型: Swin Transformer + SegFormer Head"
echo "方法: 半监督 + 对比学习 + 不确定性估计"
echo "特性:"
echo "  ✓ 监督学习"
echo "  ✓ 伪标签学习（不确定性感知）"
echo "  ✓ 一致性正则化"
echo "  ✓ 对比学习"
echo "  ✓ Monte Carlo Dropout"
echo "  ✓ 自适应阈值"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "脚本: train_swin_uncertainty.py"
echo "=========================================="
echo ""

# 8. 开始训练
echo "🎬 开始训练..."
echo ""

python train_swin_uncertainty.py

# 9. 训练完成
echo ""
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "🎉 训练完成！"
    echo "=========================================="
    echo ""
    echo "📁 结果保存在: outputs/swin_semi_uncertainty_*"
    echo ""
    echo "下一步:"
    echo "  1. 查看 TensorBoard: tensorboard --logdir=outputs"
    echo "  2. 测试模型: python test_all_checkpoints.py outputs/swin_semi_uncertainty_*"
    echo "  3. 分析不确定性: 查看 TensorBoard 中的 uncertainty_* 指标"
    echo ""
else
    echo "=========================================="
    echo "❌ 训练失败"
    echo "=========================================="
    echo ""
    echo "请检查:"
    echo "  1. 错误日志"
    echo "  2. GPU 内存"
    echo "  3. 数据路径"
    echo ""
    exit 1
fi

