#!/bin/bash
# Swin Transformer + 对比学习训练启动脚本
# 使用方法: bash start_contrastive_training.sh [gpu_id]

echo "=========================================="
echo "🚀 启动 Swin + 对比学习训练"
echo "=========================================="

# 激活swin环境
echo "📦 激活conda环境: swin"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swin

# 设置GPU（默认使用GPU 0）
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "🎮 使用GPU: $CUDA_VISIBLE_DEVICES"

# 进入项目目录
cd /home/guo_rq/segwork/nnunetplus/base_1_2

# 检查必要文件
if [ ! -f "train_swin_contrastive.py" ]; then
    echo "❌ 错误: 训练脚本不存在"
    exit 1
fi

if [ ! -f "configs/semi_supervised/ems_swin_semi_contrastive.py" ]; then
    echo "❌ 错误: 配置文件不存在"
    exit 1
fi

# 显示配置信息
echo "=========================================="
echo "📋 配置信息"
echo "=========================================="
echo "配置文件: configs/semi_supervised/ems_swin_semi_contrastive.py"
echo "训练脚本: train_swin_contrastive.py"
echo "对比学习: ✅ 启用"
echo "=========================================="
echo ""

# 询问是否继续
read -p "🤔 是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ 训练已取消"
    exit 1
fi

echo "=========================================="
echo "⏰ 开始时间: $(date)"
echo "=========================================="

# 启动训练
python train_swin_contrastive.py

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "⏰ 结束时间: $(date)"
echo "=========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "📁 结果保存在: outputs/"
    echo ""
    echo "查看结果："
    echo "  tensorboard --logdir outputs/ --port 6006"
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

