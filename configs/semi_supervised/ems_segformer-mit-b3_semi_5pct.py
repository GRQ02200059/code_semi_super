_base_ = [
    "../models/segformer_mit-b3.py",
    "../datasets/ems.py",
]

# 实验名称 - 5%标注数据
name = "segformer-mit-b3_semi_supervised_5pct_50ep"

# 训练配置
trainer = dict(
    max_epochs=50,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    check_val_every_n_epoch=1,
)

# 半监督数据配置 - 5%标注数据
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=4,
    unlabeled_batch_size=8,        # 更多无标注数据
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.05,            # 5%的数据有标注
    consistency_augmentation=True,
    seed=42,
)

# 半监督学习参数 - 针对少量标注数据优化
semi_supervised = dict(
    pseudo_threshold=0.98,         # 更高的置信度阈值
    consistency_weight=2.0,        # 更强的一致性约束
    pseudo_weight=1.5,             # 较强的伪标签学习
    ramp_up_epochs=15,             # 更长的渐进期
    ema_decay=0.999,              # 更慢的EMA更新
    use_ema_teacher=True,
)

# 损失函数
loss = "bce"

# 其他配置
mask_lc = False

# 评估配置
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)








