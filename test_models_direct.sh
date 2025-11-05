#!/bin/bash

# 自动收集所有ckpt文件
MODELS=($(find outputs/swin_semi_supervised_improved_20251104_211619/version_0/ -name "*.ckpt"))

# 安装缺失的包
pip install loguru

# 测试每个模型
for model in "${MODELS[@]}"; do
  echo "======================================================="
  echo "测试模型: $model"
  echo "======================================================="
  
  # 获取实验目录
  exp_dir=$(dirname $(dirname "$model"))
  
  # 运行测试命令
  python tools/launch.py test -e "$exp_dir" -c "$model"
  
  echo "测试完成: $model"
  echo ""
done 