#!/bin/bash
# 快速启动示例

echo "🔍 当前环境检查..."
echo "当前目录: $(pwd)"
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "📋 可选的训练命令："
echo ""
echo "# 1. 快速测试训练（10轮，约30分钟）"
echo "bash start_training.sh configs/single/ems_segformer-mit-b3_single_10ep.py"
echo ""
echo "# 2. 半监督训练（推荐，100轮）"
echo "bash start_training.sh configs/semi_supervised/ems_segformer-mit-b3_semi_improved.py"
echo ""
echo "# 3. 后台运行半监督训练"
echo "nohup bash start_training.sh configs/semi_supervised/ems_segformer-mit-b3_semi_improved.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo ""
echo "# 4. 查看训练日志"
echo "tail -f logs/training_*.log"
echo ""
