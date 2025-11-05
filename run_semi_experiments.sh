#!/bin/bash

# 半监督学习批量实验脚本
# 自动运行不同配置的半监督学习实验

echo "开始半监督学习批量实验..."
echo "时间: $(date)"

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 实验配置列表
configs=(
    "configs/semi_supervised/ems_segformer-mit-b3_semi_5pct.py"
    "configs/semi_supervised/ems_segformer-mit-b3_semi_50ep.py"
    "configs/semi_supervised/ems_upernet-rn50_semi_10pct.py"
)

# 运行每个实验
for config in "${configs[@]}"; do
    echo ""
    echo "=========================================="
    echo "运行实验: $config"
    echo "开始时间: $(date)"
    echo "=========================================="
    
    # 使用launch.py运行实验
    python tools/launch.py train -c "$config"
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "实验完成: $config"
    else
        echo "实验失败: $config"
        # 可以选择继续或停止
        # exit 1  # 取消注释以在失败时停止
    fi
    
    echo "结束时间: $(date)"
    echo ""
done

echo "所有半监督学习实验完成!"
echo "结束时间: $(date)"

# 可选: 发送通知邮件或其他通知
# echo "半监督学习实验完成" | mail -s "实验通知" your_email@example.com








