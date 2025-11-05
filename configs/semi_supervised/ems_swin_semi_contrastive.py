"""
Swin Transformer + 半监督学习 + 对比学习配置文件

集成三重学习策略:
1. 监督学习 (Supervised Learning)
2. 半监督学习 (Pseudo-labeling + Consistency Regularization)
3. 对比学习 (Contrastive Learning) ⭐ 新增

对比学习增强特征表示，提升模型区分能力
"""

_base_ = [
    "../models/segformer_swinsmall.py",
    "../datasets/ems.py",
]

# 实验名称
name = "swin_semi_contrastive"

# 训练配置
trainer = dict(
    max_epochs=100,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
)

# 早停配置
early_stopping = dict(
    monitor="val_iou",
    patience=25,  # 对比学习可能需要更多epoch收敛
    mode="max",
    verbose=True,
)

# 半监督数据配置
data = dict(
    root="data/ems",
    patch_size=512,
    modalities=["S2L2A", "DEL", "ESA_LC", "CM"],
    labeled_batch_size=6,      # 有标注数据批次大小
    unlabeled_batch_size=6,    # 无标注数据批次大小  
    batch_size_eval=8,
    num_workers=8,
    labeled_ratio=0.2,         # 20%的标注数据
    consistency_augmentation=True,  # 启用一致性增强（强弱增强）
    seed=42,
)

# 半监督学习参数
semi_supervised = dict(
    # 伪标签参数
    pseudo_threshold=0.9,      # 伪标签置信度阈值
    pseudo_weight=1.5,         # 伪标签损失权重
    
    # 一致性正则化参数
    consistency_weight=2.0,    # 一致性损失权重
    
    # EMA教师模型参数
    ema_decay=0.999,          # EMA衰减率
    use_ema_teacher=True,     # 使用教师模型
    
    # 渐进式学习参数
    ramp_up_epochs=10,        # 权重渐进增加的轮数
    
    # ⭐ 对比学习参数 (新增)
    use_contrastive=True,          # 启用对比学习
    contrastive_weight=0.5,        # 对比学习损失权重
    contrastive_temperature=0.07,  # 温度参数（控制分布平滑度）
    contrastive_mode="global",     # 对比模式: 'global' 或 'pixel'
    projection_dim=128,            # 投影头输出维度
    projection_hidden_dim=256,     # 投影头隐藏层维度
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

# 优化器配置（可选，使用默认值）
# optimizer = dict(
#     type='AdamW',
#     lr=1e-4,
#     weight_decay=1e-4,
# )

# 学习率调度器（可选）
# lr_scheduler = dict(
#     type='CosineAnnealingLR',
#     T_max=100,
#     eta_min=1e-6,
# )


# ============================================================
# 配置说明
# ============================================================
"""
对比学习参数详解:

1. use_contrastive (bool):
   - 是否启用对比学习
   - 建议: True (推荐启用)

2. contrastive_weight (float):
   - 对比学习损失的权重
   - 范围: 0.1 - 1.0
   - 建议: 0.5 (平衡对比学习和其他损失)
   - 过高: 可能影响分割性能
   - 过低: 对比学习效果不明显

3. contrastive_temperature (float):
   - 温度参数，控制相似度分布的平滑程度
   - 范围: 0.05 - 0.1
   - 建议: 0.07 (标准值)
   - 越小: 分布越尖锐，区分越严格
   - 越大: 分布越平滑，区分越宽松

4. contrastive_mode (str):
   - 'global': 全局特征对比（推荐）
     * 在图像级别进行对比
     * 计算效率高
     * 适合大部分场景
   
   - 'pixel': 像素级特征对比
     * 在像素级别进行对比
     * 计算量大，需要更多内存
     * 适合需要精细分割的场景

5. projection_dim (int):
   - 投影头输出维度
   - 建议: 128 (标准值)
   - 影响对比学习的特征表示能力

6. projection_hidden_dim (int):
   - 投影头隐藏层维度
   - 建议: 256 (是projection_dim的2倍)

============================================================
实验建议:

基础配置 (推荐新手):
  - contrastive_weight: 0.3
  - contrastive_mode: 'global'
  - 其他参数使用默认值

进阶配置 (追求性能):
  - contrastive_weight: 0.5
  - contrastive_temperature: 0.05
  - contrastive_mode: 'global'

像素级对比 (精细分割):
  - contrastive_mode: 'pixel'
  - contrastive_weight: 0.3
  - 注意: 需要更多GPU内存

============================================================
预期效果:

相比普通半监督学习:
  + 特征表示能力 ↑ 10-15%
  + IoU提升 ↑ 1-3%
  + 训练稳定性 ↑
  + 对小样本数据更鲁棒

代价:
  - 训练时间 ↑ 15-20%
  - GPU内存 ↑ 10-15%
  - 模型参数 ↑ (增加投影头)

============================================================
"""

