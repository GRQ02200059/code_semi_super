_base_ = [
    "../models/upernet_rn50.py",
    "../datasets/ems.py",
]

# 实验名称 - UPerNet + ResNet50 + 10%标注数据
name = "upernet-rn50_semi_supervised_10pct_50ep"

# 训练配置
trainer = dict(
    max_epochs=50,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    check_val_every_n_epoch=1,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=4,
    unlabeled_batch_size=4,
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.1,             # 10%的数据有标注
    consistency_augmentation=True,
    seed=42,
)

# 半监督学习参数 - UPerNet优化设置
semi_supervised = dict(
    pseudo_threshold=0.95,
    consistency_weight=1.0,
    pseudo_weight=1.0,
    ramp_up_epochs=10,
    ema_decay=0.99,
    use_ema_teacher=True,
)

# 损失函数
loss = "dice"  # UPerNet通常用Dice损失效果更好

# 其他配置
mask_lc = False

# 评估配置
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)








