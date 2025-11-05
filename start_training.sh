#!/bin/bash
# 快速启动训练脚本
# 使用方法: bash start_training.sh [config_path]

echo "=========================================="
echo "🚀 启动训练"
echo "=========================================="

# 激活swin环境
echo "激活conda环境: swin"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swin

# 设置GPU（默认使用GPU 0）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 进入项目目录
cd /home/guo_rq/segwork/nnunetplus/base_1_2

# 获取配置文件（默认使用半监督训练）
CONFIG=${1:-"configs/semi_supervised/ems_segformer-mit-b3_semi_improved.py"}
echo "配置文件: $CONFIG"

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi

echo "=========================================="
echo "开始时间: $(date)"
echo "=========================================="

# 启动训练
python tools/launch.py train -c "$CONFIG"

echo "=========================================="
echo "结束时间: $(date)"
echo "=========================================="

