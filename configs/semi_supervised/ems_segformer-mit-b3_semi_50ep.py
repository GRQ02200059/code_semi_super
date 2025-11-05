_base_ = [
    "../models/segformer_mit-b3.py",
    "../datasets/ems.py",
]

# 实验名称
name = "segformer-mit-b3_semi_supervised_50ep"

# 训练配置
trainer = dict(
    max_epochs=50,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    # 半监督学习通常需要更频繁的验证
    check_val_every_n_epoch=1,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=4,      # 有标注数据批次大小
    unlabeled_batch_size=4,    # 无标注数据批次大小  
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.1,         # 10%的数据有标注
    consistency_augmentation=True,  # 启用一致性增强
    seed=42,
)

# 半监督学习参数
semi_supervised = dict(
    pseudo_threshold=0.95,     # 伪标签置信度阈值
    consistency_weight=1.0,    # 一致性损失权重
    pseudo_weight=1.0,         # 伪标签损失权重
    ramp_up_epochs=10,         # 权重渐进增加的轮数
    ema_decay=0.99,           # EMA教师模型衰减率
    use_ema_teacher=True,     # 使用EMA教师模型
)

# 损失函数
loss = "bce"  # 可以选择 "bce" 或 "dice"

# 其他配置
mask_lc = False

# 评估配置
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)








