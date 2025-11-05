"""
增强版半监督学习模块 - 集成对比学习
支持伪标签、一致性正则化和对比学习的三重学习策略
"""

from typing import Any, Callable, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from loguru import logger as log

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss, SupConLoss, PixelContrastLoss, ProjectionHead
from baseg.modules.semi_supervised import SemiSupervisedModule


class SemiSupervisedContrastiveModule(SemiSupervisedModule):
    """
    集成对比学习的半监督学习模块
    
    特性:
    1. 伪标签生成 (Pseudo-Labeling)
    2. 一致性正则化 (Consistency Regularization)
    3. 对比学习 (Contrastive Learning) ⭐ 新增
    4. EMA教师模型 (Mean Teacher)
    
    对比学习增强了特征表示能力，使模型能够更好地区分不同类别
    """
    
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
        # 半监督学习参数
        pseudo_threshold: float = 0.95,
        consistency_weight: float = 1.0,
        pseudo_weight: float = 1.0,
        ramp_up_epochs: int = 10,
        ema_decay: float = 0.99,
        use_ema_teacher: bool = True,
        # 对比学习参数 ⭐ 新增
        use_contrastive: bool = True,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.07,
        contrastive_mode: str = "global",  # 'global' 或 'pixel'
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
    ):
        """
        Args:
            config: 模型配置
            ... (其他参数与SemiSupervisedModule相同)
            use_contrastive: 是否启用对比学习
            contrastive_weight: 对比学习损失权重
            contrastive_temperature: 对比学习温度参数
            contrastive_mode: 'global' - 全局特征对比, 'pixel' - 像素级对比
            projection_dim: 投影头输出维度
            projection_hidden_dim: 投影头隐藏层维度
        """
        # 调用父类初始化
        super().__init__(
            config, tiler, predict_callback, loss,
            pseudo_threshold, consistency_weight, pseudo_weight,
            ramp_up_epochs, ema_decay, use_ema_teacher
        )
        
        # 对比学习参数
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_mode = contrastive_mode
        
        if self.use_contrastive:
            # 获取backbone输出维度
            backbone_out_dim = self._get_backbone_out_dim()
            
            # 创建投影头
            self.projection_head = ProjectionHead(
                in_dim=backbone_out_dim,
                hidden_dim=projection_hidden_dim,
                out_dim=projection_dim,
                num_layers=2,
                use_bn=True
            )
            
            # 创建对比学习损失函数
            if contrastive_mode == "global":
                self.contrastive_loss_fn = SupConLoss(
                    temperature=contrastive_temperature
                )
            elif contrastive_mode == "pixel":
                self.contrastive_loss_fn = PixelContrastLoss(
                    temperature=contrastive_temperature,
                    max_samples=1024
                )
            else:
                raise ValueError(f"Unknown contrastive_mode: {contrastive_mode}")
            
            log.info(f"✅ 对比学习已启用: mode={contrastive_mode}, "
                    f"weight={contrastive_weight}, temp={contrastive_temperature}")
        else:
            log.info("❌ 对比学习未启用")
    
    def _get_backbone_out_dim(self) -> int:
        """获取backbone输出特征的维度
        
        对于Swin Transformer，通常最后一层的输出维度
        """
        # 尝试从模型配置中获取
        if hasattr(self.model, 'backbone'):
            # 对于Swin Transformer
            if hasattr(self.model.backbone, 'embed_dims'):
                # Swin的最后一层维度是 embed_dims * 2^(num_stages-1)
                base_dim = self.model.backbone.embed_dims
                # Swin-Small: 96 -> 192 -> 384 -> 768
                num_stages = len(self.model.backbone.depths)
                return base_dim * (2 ** (num_stages - 1))
            # 对于其他backbone，尝试直接获取
            elif hasattr(self.model.backbone, 'out_channels'):
                return self.model.backbone.out_channels[-1]
        
        # 默认值（Swin-Small的Stage4输出）
        default_dim = 768
        log.warning(f"⚠️  无法自动获取backbone维度，使用默认值: {default_dim}")
        return default_dim
    
    def _extract_features(
        self, 
        images: torch.Tensor, 
        use_projection: bool = True
    ) -> torch.Tensor:
        """提取特征用于对比学习
        
        Args:
            images: [B, C, H, W] 输入图像
            use_projection: 是否使用投影头
            
        Returns:
            features: [B, D] 或 [B, D, H', W'] 特征
        """
        # 提取backbone特征
        if hasattr(self.model, 'backbone'):
            # 获取最后一层特征
            backbone_features = self.model.backbone(images)
            if isinstance(backbone_features, (list, tuple)):
                # 使用最后一层特征
                features = backbone_features[-1]
            else:
                features = backbone_features
        else:
            # 如果没有backbone属性，使用整个模型
            features = self.model.encode(images) if hasattr(self.model, 'encode') else self.model(images)
        
        # 全局模式：进行全局平均池化
        if self.contrastive_mode == "global" and use_projection:
            # [B, C, H, W] -> [B, C]
            if features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten(1)
            
            # 通过投影头
            if hasattr(self, 'projection_head'):
                features = self.projection_head(features)
        
        return features
    
    def _compute_contrastive_loss(
        self, 
        images_labeled: torch.Tensor,
        images_unlabeled_weak: torch.Tensor,
        images_unlabeled_strong: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """计算对比学习损失
        
        Args:
            images_labeled: [B1, C, H, W] 有标注图像
            images_unlabeled_weak: [B2, C, H, W] 无标注图像（弱增强）
            images_unlabeled_strong: [B2, C, H, W] 无标注图像（强增强）
            labels: [B1, H, W] 标注（可选，用于像素级对比）
            
        Returns:
            contrastive_loss: 对比损失
        """
        if self.contrastive_mode == "global":
            # 全局特征对比
            return self._compute_global_contrastive_loss(
                images_labeled, 
                images_unlabeled_weak,
                images_unlabeled_strong
            )
        elif self.contrastive_mode == "pixel":
            # 像素级特征对比
            return self._compute_pixel_contrastive_loss(
                images_labeled,
                labels
            )
        else:
            return torch.tensor(0.0, device=images_labeled.device)
    
    def _compute_global_contrastive_loss(
        self,
        images_labeled: torch.Tensor,
        images_unlabeled_weak: torch.Tensor,
        images_unlabeled_strong: torch.Tensor
    ) -> torch.Tensor:
        """计算全局对比学习损失
        
        策略:
        1. 有标注数据：原图作为一个view
        2. 无标注数据：弱增强和强增强作为两个view（正样本对）
        3. 不同图像之间作为负样本对
        """
        # 提取特征
        features_labeled = self._extract_features(images_labeled, use_projection=True)
        features_unlabeled_weak = self._extract_features(images_unlabeled_weak, use_projection=True)
        features_unlabeled_strong = self._extract_features(images_unlabeled_strong, use_projection=True)
        
        # 组合所有特征
        # 有标注数据：[B1, D]
        # 无标注数据：[B2, 2, D] (weak + strong)
        batch_size_labeled = features_labeled.shape[0]
        batch_size_unlabeled = features_unlabeled_weak.shape[0]
        
        # 将无标注数据的两个view堆叠
        features_unlabeled = torch.stack([
            features_unlabeled_weak,
            features_unlabeled_strong
        ], dim=1)  # [B2, 2, D]
        
        # 有标注数据只有一个view
        features_labeled = features_labeled.unsqueeze(1)  # [B1, 1, D]
        
        # 合并所有特征
        all_features = torch.cat([
            features_labeled,
            features_unlabeled
        ], dim=0)  # [B1+B2, n_views, D]
        
        # 计算对比损失（无监督模式，仅基于数据增强）
        contrastive_loss = self.contrastive_loss_fn(all_features)
        
        return contrastive_loss
    
    def _compute_pixel_contrastive_loss(
        self,
        images_labeled: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算像素级对比学习损失
        
        仅在有标注数据上计算，相同类别的像素作为正样本对
        """
        # 提取特征图（不使用投影头，保持空间维度）
        features = self._extract_features(images_labeled, use_projection=False)
        
        # 计算像素级对比损失
        contrastive_loss = self.contrastive_loss_fn(features, labels)
        
        return contrastive_loss
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - 增强版，集成对比学习"""
        # 检查批次格式
        if isinstance(batch, dict) and 'labeled' in batch and 'unlabeled' in batch:
            # 半监督数据格式
            return self._training_step_semi_supervised_with_contrastive(batch, batch_idx)
        else:
            # 标准监督学习格式
            return super().training_step(batch, batch_idx)
    
    def _training_step_semi_supervised_with_contrastive(self, batch, batch_idx):
        """半监督训练步骤（集成对比学习）"""
        labeled_data = batch['labeled']
        unlabeled_data = batch['unlabeled']
        
        if labeled_data is None and unlabeled_data is None:
            return None
        
        total_loss = 0.0
        losses_dict = {}
        
        # 获取当前epoch的权重
        current_epoch = self.current_epoch
        consistency_weight, pseudo_weight = self._get_current_weights(current_epoch)
        
        # 对比学习权重也采用渐进式增加
        if current_epoch < self.ramp_up_epochs:
            ramp_up_factor = current_epoch / self.ramp_up_epochs
        else:
            ramp_up_factor = 1.0
        current_contrastive_weight = self.contrastive_weight * ramp_up_factor
        
        # ==================== 1. 监督学习损失 ====================
        if labeled_data is not None:
            x_labeled = labeled_data['image']
            y_labeled = labeled_data['mask_del']
            
            # 前向传播
            logits_labeled = self.model(x_labeled)
            if logits_labeled.dim() == 4:
                logits_labeled = logits_labeled.squeeze(1)
            
            # 监督损失
            supervised_loss = self.criterion_supervised(logits_labeled, y_labeled.float())
            total_loss += supervised_loss
            losses_dict['supervised_loss'] = supervised_loss.item()
            
            # 计算指标
            with torch.no_grad():
                pred_labeled = (torch.sigmoid(logits_labeled) > 0.5).float()
                self._update_metrics(pred_labeled, y_labeled, prefix='train')
        
        # ==================== 2. 对比学习损失 ⭐ 新增 ====================
        if self.use_contrastive and unlabeled_data is not None and labeled_data is not None:
            x_labeled = labeled_data['image']
            y_labeled = labeled_data['mask_del']
            
            # 无标注数据
            if 'image_weak' in unlabeled_data and 'image_strong' in unlabeled_data:
                x_unlabeled_weak = unlabeled_data['image_weak']
                x_unlabeled_strong = unlabeled_data['image_strong']
            else:
                # 如果没有分开的弱强增强，使用同一个
                x_unlabeled_weak = unlabeled_data['image']
                x_unlabeled_strong = unlabeled_data['image']
            
            # 计算对比学习损失
            try:
                contrastive_loss = self._compute_contrastive_loss(
                    x_labeled,
                    x_unlabeled_weak,
                    x_unlabeled_strong,
                    y_labeled
                )
                
                weighted_contrastive_loss = current_contrastive_weight * contrastive_loss
                total_loss += weighted_contrastive_loss
                losses_dict['contrastive_loss'] = contrastive_loss.item()
                losses_dict['weighted_contrastive_loss'] = weighted_contrastive_loss.item()
            except Exception as e:
                log.warning(f"⚠️  对比学习损失计算失败: {e}")
                losses_dict['contrastive_loss'] = 0.0
        
        # ==================== 3. 伪标签损失 ====================
        if unlabeled_data is not None and pseudo_weight > 0:
            # 使用弱增强图像生成伪标签
            if 'image_weak' in unlabeled_data:
                x_unlabeled_weak = unlabeled_data['image_weak']
            else:
                x_unlabeled_weak = unlabeled_data['image']
            
            # 生成伪标签
            pseudo_labels, confidence_mask = self._generate_pseudo_labels(
                x_unlabeled_weak, 
                use_teacher=self.use_ema_teacher
            )
            
            # 使用强增强图像进行训练
            if 'image_strong' in unlabeled_data:
                x_unlabeled_strong = unlabeled_data['image_strong']
            else:
                x_unlabeled_strong = unlabeled_data['image']
            
            logits_unlabeled = self.model(x_unlabeled_strong)
            if logits_unlabeled.dim() == 4:
                logits_unlabeled = logits_unlabeled.squeeze(1)
            
            # 仅在高置信度区域计算损失
            if confidence_mask.sum() > 0:
                pseudo_loss = self.criterion_unsupervised(
                    logits_unlabeled, 
                    pseudo_labels.float()
                )
                # 应用置信度掩码
                pseudo_loss = (pseudo_loss * confidence_mask).sum() / (confidence_mask.sum() + 1e-6)
                
                weighted_pseudo_loss = pseudo_weight * pseudo_loss
                total_loss += weighted_pseudo_loss
                losses_dict['pseudo_loss'] = pseudo_loss.item()
                losses_dict['pseudo_weight'] = pseudo_weight
                
                # 统计伪标签信息
                self.pseudo_label_stats['total_pixels'] += pseudo_labels.numel()
                self.pseudo_label_stats['pseudo_pixels'] += confidence_mask.sum().item()
                self.pseudo_label_stats['positive_pseudo'] += (pseudo_labels * confidence_mask).sum().item()
        
        # ==================== 4. 一致性损失 ====================
        if unlabeled_data is not None and consistency_weight > 0:
            if 'image_weak' in unlabeled_data and 'image_strong' in unlabeled_data:
                x_unlabeled_weak = unlabeled_data['image_weak']
                x_unlabeled_strong = unlabeled_data['image_strong']
                
                # 弱增强预测
                with torch.no_grad():
                    logits_weak = self.model(x_unlabeled_weak)
                    if logits_weak.dim() == 4:
                        logits_weak = logits_weak.squeeze(1)
                
                # 强增强预测
                logits_strong = self.model(x_unlabeled_strong)
                if logits_strong.dim() == 4:
                    logits_strong = logits_strong.squeeze(1)
                
                # 计算一致性损失
                consistency_loss = self._compute_consistency_loss(logits_weak, logits_strong)
                
                weighted_consistency_loss = consistency_weight * consistency_loss
                total_loss += weighted_consistency_loss
                losses_dict['consistency_loss'] = consistency_loss.item()
                losses_dict['consistency_weight'] = consistency_weight
        
        # 记录总损失
        losses_dict['train_loss'] = total_loss.item()
        losses_dict['contrastive_weight'] = current_contrastive_weight
        
        # 日志记录
        for key, value in losses_dict.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """训练批次结束时的回调"""
        # 更新EMA教师模型
        if self.use_ema_teacher:
            self._update_teacher_model()
        
        # 调用父类方法
        super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        # 记录伪标签统计信息
        if self.pseudo_label_stats['total_pixels'] > 0:
            pseudo_ratio = self.pseudo_label_stats['pseudo_pixels'] / self.pseudo_label_stats['total_pixels']
            self.log('pseudo_label_ratio', pseudo_ratio, prog_bar=True)
            
            if self.pseudo_label_stats['pseudo_pixels'] > 0:
                positive_ratio = self.pseudo_label_stats['positive_pseudo'] / self.pseudo_label_stats['pseudo_pixels']
                self.log('pseudo_positive_ratio', positive_ratio)
        
        # 重置统计
        self.pseudo_label_stats = {
            'total_pixels': 0,
            'pseudo_pixels': 0,
            'positive_pseudo': 0
        }
        
        # 计算并记录训练指标（使用.items()遍历）
        train_metrics_dict = {}
        for name, metric in self.train_metrics.items():
            value = metric.compute()
            train_metrics_dict[name] = value
            self.log(f'train_{name}', value, prog_bar=True)
        
        # 重置训练指标
        for metric in self.train_metrics.values():
            metric.reset()
        
        # 日志
        current_epoch = self.current_epoch
        train_f1 = train_metrics_dict.get('f1', 0.0)
        train_iou = train_metrics_dict.get('iou', 0.0)
        
        log.info(f"半监督+对比学习 Epoch {current_epoch} 结束. "
                f"Metrics: train_f1={train_f1:.4f}, train_iou={train_iou:.4f}")


__all__ = ['SemiSupervisedContrastiveModule']

