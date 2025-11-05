_base_ = [
    "../models/segformer_mit-b3.py",
    "../datasets/ems.py",
]

# 实验名称
name = "segformer-mit-b3_semi_supervised_pretrained"

# 训练配置
trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    check_val_every_n_epoch=1,
)

# 早停配置
early_stopping = dict(
    monitor="val_iou",
    patience=15,
    mode="max",
    verbose=True,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=8,
    unlabeled_batch_size=8,
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.3,         # 进一步增加到30%的标注数据
    consistency_augmentation=True,
    seed=42,
)

# 半监督学习参数
semi_supervised = dict(
    pseudo_threshold=0.85,     # 进一步降低阈值
    consistency_weight=3.0,    # 更强的正则化
    pseudo_weight=2.0,         # 更强的伪标签学习
    ramp_up_epochs=5,          # 更快的权重增长
    ema_decay=0.9995,         # 更稳定的教师模型
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






