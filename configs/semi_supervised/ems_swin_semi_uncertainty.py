"""
Swin Transformer + 半监督学习 + 对比学习 + 不确定性估计配置文件

集成五重学习策略:
1. 监督学习 (Supervised Learning)
2. 半监督学习 (Pseudo-labeling + Consistency Regularization)
3. 对比学习 (Contrastive Learning)
4. 不确定性估计 (Uncertainty Estimation) ⭐ 新增

不确定性估计通过Monte Carlo Dropout动态调整伪标签阈值，提高伪标签质量
"""

_base_ = [
    "../models/segformer_swinsmall.py",
    "../datasets/ems.py",
]

# 实验名称
name = "swin_semi_uncertainty"

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
    patience=25,
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
    consistency_augmentation=True,  # 启用一致性增强
    seed=42,
)

# 半监督学习参数
semi_supervised = dict(
    # 伪标签参数
    pseudo_threshold=0.9,      # 基础伪标签置信度阈值
    pseudo_weight=1.5,         # 伪标签损失权重
    
    # 一致性正则化参数
    consistency_weight=2.0,    # 一致性损失权重
    
    # EMA教师模型参数
    ema_decay=0.999,          # EMA衰减率
    use_ema_teacher=True,     # 使用教师模型
    
    # 渐进式学习参数
    ramp_up_epochs=10,        # 权重渐进增加的轮数
    
    # 对比学习参数
    use_contrastive=True,          # 启用对比学习
    contrastive_weight=0.5,        # 对比学习损失权重
    contrastive_temperature=0.07,  # 温度参数
    contrastive_mode="global",     # 对比模式: 'global' 或 'pixel'
    projection_dim=128,            # 投影头输出维度
    projection_hidden_dim=256,     # 投影头隐藏层维度
    
    # ⭐ 不确定性估计参数 (新增)
    use_uncertainty=True,              # 启用不确定性估计
    uncertainty_n_samples=5,           # Monte Carlo采样次数
    uncertainty_weight=0.1,            # 不确定性对阈值的影响权重
    min_threshold=0.8,                 # 最小自适应阈值
    max_threshold=0.99,                # 最大自适应阈值
    use_adaptive_threshold=True,       # 使用自适应阈值
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


# ============================================================
# 配置说明
# ============================================================
"""
不确定性估计参数详解:

1. use_uncertainty (bool):
   - 是否启用不确定性估计
   - 建议: True (强烈推荐)
   - 效果: 动态调整伪标签阈值，提高伪标签质量

2. uncertainty_n_samples (int):
   - Monte Carlo Dropout采样次数
   - 范围: 3-10
   - 建议: 5 (平衡精度和速度)
   - 越大越准确但速度越慢

3. uncertainty_weight (float):
   - 不确定性对阈值的影响权重
   - 范围: 0.05-0.2
   - 建议: 0.1 (标准值)
   - 含义: threshold = base_threshold + uncertainty * weight

4. use_adaptive_threshold (bool):
   - 是否使用自适应阈值
   - True: 根据不确定性动态调整阈值
   - False: 使用固定的pseudo_threshold
   - 建议: True

5. min_threshold / max_threshold (float):
   - 自适应阈值的范围限制
   - 建议: [0.8, 0.99]
   - 防止阈值过低或过高

============================================================
工作原理:

1. Monte Carlo Dropout:
   - 多次前向传播（启用dropout）
   - 计算预测的均值和方差
   - 方差越大 = 不确定性越高

2. 自适应阈值:
   - 不确定性低的区域: 使用较低阈值（接受更多伪标签）
   - 不确定性高的区域: 使用较高阈值（更保守）

3. 伪标签生成:
   ```
   uncertainty = var(predictions)  # 预测方差
   adaptive_threshold = base_threshold + uncertainty * weight
   adaptive_threshold = clamp(adaptive_threshold, min_thresh, max_thresh)
   
   confidence = max(prob, 1-prob)
   mask = confidence > adaptive_threshold
   ```

============================================================
实验建议:

基础配置 (推荐新手):
  - use_uncertainty: True
  - uncertainty_n_samples: 5
  - use_adaptive_threshold: True
  - 其他参数使用默认值

高精度配置:
  - uncertainty_n_samples: 10
  - uncertainty_weight: 0.15
  - min_threshold: 0.85

快速配置 (训练速度优先):
  - uncertainty_n_samples: 3
  - uncertainty_weight: 0.08

对比实验:
  1. 无不确定性: use_uncertainty=False
  2. 固定阈值: use_adaptive_threshold=False
  3. 完整配置: 两者都启用

============================================================
预期效果:

相比标准半监督:
  + 伪标签质量 ↑ 15-25%
  + IoU提升 ↑ 1-2%
  + 训练稳定性 ↑
  + 适应性更强

相比对比学习版本:
  + 伪标签精度 ↑ 10-15%
  + IoU额外提升 ↑ 0.5-1%
  
代价:
  - 训练时间 ↑ 10-20% (Monte Carlo采样)
  - GPU内存 ↑ 5-10%

============================================================
监控指标:

TensorBoard中新增:
  - mean_uncertainty: 平均不确定性
  - max_uncertainty: 最大不确定性
  - uncertainty_adaptive_threshold_mean: 自适应阈值均值
  
分析:
  - mean_uncertainty越小越好（模型越确定）
  - 训练初期不确定性高，后期应该降低
  - 自适应阈值应该在[min, max]范围内波动

============================================================
"""

