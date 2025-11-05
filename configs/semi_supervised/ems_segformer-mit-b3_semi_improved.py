_base_ = [
    "../models/segformer_mit-b3.py",
    "../datasets/ems.py",
]

# 实验名称
name = "segformer-mit-b3_semi_supervised_improved"

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
    monitor="val_iou",     # 监控IoU而不是loss
    patience=20,           # 减少patience
    mode="max",            # IoU越大越好
    verbose=True,
)

# 改进的半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=6,      # 增加有标注数据批次大小
    unlabeled_batch_size=6,    # 增加无标注数据批次大小  
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.2,         # 增加到20%的标注数据
    consistency_augmentation=True,
    seed=42,
)

# 改进的半监督学习参数
semi_supervised = dict(
    pseudo_threshold=0.9,      # 降低伪标签置信度阈值
    consistency_weight=2.0,    # 增加一致性损失权重
    pseudo_weight=1.5,         # 增加伪标签损失权重
    ramp_up_epochs=10,         # 减少权重渐进增加的轮数
    ema_decay=0.999,          # 增加EMA衰减率，使教师模型更稳定
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






