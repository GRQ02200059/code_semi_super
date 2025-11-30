_base_ = [
    "../models/segformer_swinsmall.py",
    "../datasets/ems.py",
]

# 实验名称
name = "swin_semi_multiscale_test"

# 训练配置（快速测试版）
trainer = dict(
    max_epochs=10,  # 快速测试，只训练10个epoch
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    check_val_every_n_epoch=1,
)

# 早停配置
early_stopping = dict(
    monitor="val_iou",
    patience=5,
    mode="max",
    verbose=True,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=4,      # 减小batch size以便快速测试
    unlabeled_batch_size=4,
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.1,         # 使用10%标注数据
    consistency_augmentation=True,
    seed=42,
)

# 半监督学习参数（包含多尺度一致性）
semi_supervised = dict(
    # 基础半监督参数
    pseudo_threshold=0.9,
    consistency_weight=1.0,
    pseudo_weight=1.0,
    ramp_up_epochs=5,          # 快速ramp-up
    ema_decay=0.999,
    use_ema_teacher=True,
    
    # 多尺度一致性参数 ⭐ 核心功能
    use_multiscale=True,                    # 启用多尺度一致性
    multiscale_weight=0.5,                  # 多尺度损失权重
    multiscale_scales=[0.75, 1.0, 1.25],   # 3个尺度（标准配置）
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



