_base_ = [
    "../models/segformer_swinsmall.py",
    "../datasets/ems.py",
]

# 实验名称
name = "swin_semi_multiscale_tuned_v2"

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
    patience=30,
    mode="max",
    verbose=True,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=6,
    unlabeled_batch_size=6,
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.2,
    consistency_augmentation=True,
    seed=42,
)

# 调优方案2: 更小的尺度范围 ⭐
semi_supervised = dict(
    pseudo_threshold=0.9,
    consistency_weight=2.0,
    pseudo_weight=1.5,
    ramp_up_epochs=10,
    ema_decay=0.999,
    use_ema_teacher=True,
    # 多尺度一致性 - 更小的尺度范围
    use_multiscale=True,
    multiscale_weight=0.3,                      # 适中权重
    multiscale_scales=[0.9, 1.0, 1.1],          # 更小范围 ⭐
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

