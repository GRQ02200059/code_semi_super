# 半监督学习实现指南

本项目实现了一个完整的半监督学习框架，用于遥感影像燃烧区域分割任务。

## 🎯 半监督学习方法

### 核心技术
1. **伪标签生成** (Pseudo-Labeling)
   - 使用高置信度预测作为无标注数据的伪标签
   - 可配置的置信度阈值过滤低质量伪标签
   - 支持EMA教师模型生成更稳定的伪标签

2. **一致性正则化** (Consistency Regularization)
   - 对同一图像的不同增强版本保持预测一致性
   - 使用MSE损失计算一致性约束
   - 强增强 vs 弱增强策略

3. **渐进式学习** (Progressive Learning)
   - 动态调整半监督损失权重
   - 前期专注监督学习，后期逐渐引入半监督约束
   - 可配置的权重增长策略

## 📁 文件结构

```
├── src/baseg/modules/semi_supervised.py    # 半监督学习模块
├── src/baseg/datamodules_semi.py          # 半监督数据模块
├── configs/semi_supervised/               # 半监督配置文件
├── train_semi_supervised.py               # 半监督训练脚本
└── SEMI_SUPERVISED_README.md              # 使用说明
```

## 🚀 快速开始

### 1. 基本训练
```bash
# 使用10%标注数据进行半监督训练
python train_semi_supervised.py
```

### 2. 使用launch.py训练
```bash
# 使用配置文件训练
python tools/launch.py train -c configs/semi_supervised/ems_segformer-mit-b3_semi_50ep.py
```

### 3. 自定义配置
创建新的配置文件，修改以下参数：

```python
# 数据配置
data = dict(
    labeled_ratio=0.1,         # 标注数据比例 (10%)
    labeled_batch_size=4,      # 标注数据批次大小
    unlabeled_batch_size=4,    # 无标注数据批次大小
    consistency_augmentation=True,  # 启用一致性增强
)

# 半监督参数
semi_supervised = dict(
    pseudo_threshold=0.95,     # 伪标签置信度阈值
    consistency_weight=1.0,    # 一致性损失权重
    pseudo_weight=1.0,         # 伪标签损失权重
    ramp_up_epochs=10,         # 权重渐进增加轮数
    ema_decay=0.99,           # EMA教师模型衰减率
    use_ema_teacher=True,     # 使用EMA教师模型
)
```

## ⚙️ 关键参数说明

### 数据相关参数
- `labeled_ratio`: 有标注数据的比例 (0.05-0.2)
- `consistency_augmentation`: 是否启用一致性增强
- `labeled_batch_size`: 有标注数据批次大小
- `unlabeled_batch_size`: 无标注数据批次大小

### 半监督学习参数
- `pseudo_threshold`: 伪标签置信度阈值 (0.9-0.99)
  - 越高越严格，质量更好但数量更少
- `consistency_weight`: 一致性损失权重 (0.1-2.0)
  - 控制一致性约束的强度
- `pseudo_weight`: 伪标签损失权重 (0.1-2.0)
  - 控制伪标签学习的强度
- `ramp_up_epochs`: 权重渐进增加的轮数 (5-20)
  - 前期专注监督学习，后期引入半监督
- `ema_decay`: EMA教师模型衰减率 (0.99-0.999)
  - 控制教师模型更新速度
- `use_ema_teacher`: 是否使用EMA教师模型
  - 建议开启，提供更稳定的伪标签

## 📊 实验建议

### 1. 标注数据比例实验
```bash
# 5% 标注数据
labeled_ratio=0.05

# 10% 标注数据  
labeled_ratio=0.1

# 20% 标注数据
labeled_ratio=0.2
```

### 2. 置信度阈值实验
```bash
# 高置信度，少但准确的伪标签
pseudo_threshold=0.98

# 中等置信度，平衡质量和数量
pseudo_threshold=0.95

# 低置信度，更多但可能有噪声的伪标签
pseudo_threshold=0.9
```

### 3. 损失权重实验
```bash
# 强一致性约束
consistency_weight=2.0
pseudo_weight=0.5

# 平衡设置
consistency_weight=1.0
pseudo_weight=1.0

# 强伪标签学习
consistency_weight=0.5
pseudo_weight=2.0
```

## 📈 监控指标

### TensorBoard可视化
训练过程中可以监控以下指标：

1. **损失指标**
   - `train_loss`: 总训练损失
   - `supervised_loss`: 监督学习损失
   - `consistency_loss`: 一致性损失
   - `val_loss`: 验证损失

2. **半监督指标**
   - `pseudo_label_ratio`: 伪标签使用比例
   - `pseudo_positive_ratio`: 伪标签中正样本比例
   - `consistency_weight`: 当前一致性权重
   - `pseudo_weight`: 当前伪标签权重

3. **性能指标**
   - `train_f1`: 训练F1分数
   - `train_iou`: 训练IoU
   - `val_f1`: 验证F1分数
   - `val_iou`: 验证IoU

## 🔧 故障排除

### 1. 内存不足
- 减少batch size: `labeled_batch_size=2, unlabeled_batch_size=2`
- 减少patch size: `patch_size=256`
- 关闭一致性增强: `consistency_augmentation=False`

### 2. 伪标签质量差
- 提高置信度阈值: `pseudo_threshold=0.98`
- 增加ramp-up轮数: `ramp_up_epochs=20`
- 启用EMA教师模型: `use_ema_teacher=True`

### 3. 训练不稳定
- 降低学习率: 在optimizer中设置更小的lr
- 增加EMA衰减: `ema_decay=0.999`
- 减少权重: `consistency_weight=0.5, pseudo_weight=0.5`

### 4. 收敛缓慢
- 增加有标注数据比例: `labeled_ratio=0.2`
- 提前引入半监督: `ramp_up_epochs=5`
- 增加权重: `consistency_weight=2.0, pseudo_weight=2.0`

## 🎯 最佳实践

1. **从小规模开始**: 先用5-10%的标注数据验证方法有效性
2. **渐进式调参**: 先固定大部分参数，逐个调优关键参数
3. **多次实验**: 半监督学习有随机性，建议多次实验取平均
4. **监控伪标签**: 关注伪标签的质量和使用比例
5. **对比基线**: 与纯监督学习方法对比，验证改进效果

## 📚 参考文献

1. **Pseudo-Labeling**: Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.
2. **Mean Teacher**: Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models.
3. **FixMatch**: Sohn, K., et al. (2020). Fixmatch: Simplifying semi-supervised learning with consistency and confidence.
4. **Remote Sensing SSL**: Various papers on semi-supervised learning for remote sensing applications.

## 🤝 贡献

欢迎提交Issue和Pull Request来改进半监督学习实现！








