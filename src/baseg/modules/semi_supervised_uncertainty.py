"""
集成不确定性估计的半监督学习模块
支持伪标签、一致性正则化、对比学习和不确定性估计
"""

from typing import Any, Callable, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from loguru import logger as log

from baseg.modules.semi_supervised_contrastive import SemiSupervisedContrastiveModule
from baseg.modules.uncertainty import (
    UncertaintyEstimator,
    UncertaintyAwarePseudoLabeling
)


class SemiSupervisedUncertaintyModule(SemiSupervisedContrastiveModule):
    """
    集成不确定性估计的半监督学习模块
    
    特性:
    1. 监督学习 (Supervised Learning)
    2. 伪标签学习 (Pseudo-Labeling with Uncertainty)
    3. 一致性正则化 (Consistency Regularization)
    4. 对比学习 (Contrastive Learning)
    5. 不确定性估计 (Uncertainty Estimation) ⭐ 新增
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
        # 对比学习参数
        use_contrastive: bool = True,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.07,
        contrastive_mode: str = "global",
        projection_dim: int = 128,
        projection_hidden_dim: int = 256,
        # 不确定性估计参数 ⭐ 新增
        use_uncertainty: bool = True,
        uncertainty_n_samples: int = 5,
        uncertainty_weight: float = 0.1,
        min_threshold: float = 0.8,
        max_threshold: float = 0.99,
        use_adaptive_threshold: bool = True,
    ):
        """
        Args:
            ... (其他参数与父类相同)
            use_uncertainty: 是否启用不确定性估计
            uncertainty_n_samples: Monte Carlo采样次数
            uncertainty_weight: 不确定性对阈值的影响权重
            min_threshold: 最小自适应阈值
            max_threshold: 最大自适应阈值
            use_adaptive_threshold: 是否使用自适应阈值
        """
        # 调用父类初始化
        super().__init__(
            config, tiler, predict_callback, loss,
            pseudo_threshold, consistency_weight, pseudo_weight,
            ramp_up_epochs, ema_decay, use_ema_teacher,
            use_contrastive, contrastive_weight, contrastive_temperature,
            contrastive_mode, projection_dim, projection_hidden_dim
        )
        
        # 不确定性估计参数
        self.use_uncertainty = use_uncertainty
        self.use_adaptive_threshold = use_adaptive_threshold
        
        if self.use_uncertainty:
            # 创建不确定性估计器
            self.uncertainty_estimator = UncertaintyEstimator(
                n_samples=uncertainty_n_samples,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                uncertainty_weight=uncertainty_weight
            )
            
            # 创建不确定性感知伪标签生成器
            self.uncertainty_pseudo_labeling = UncertaintyAwarePseudoLabeling(
                estimator=self.uncertainty_estimator,
                base_threshold=pseudo_threshold,
                use_adaptive_threshold=use_adaptive_threshold,
                log_uncertainty=True
            )
            
            log.info(f"✅ 不确定性估计已启用: n_samples={uncertainty_n_samples}, "
                    f"adaptive_threshold={use_adaptive_threshold}")
        else:
            log.info("❌ 不确定性估计未启用")
    
    def _generate_pseudo_labels_with_uncertainty(
        self,
        images: torch.Tensor,
        use_teacher: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        使用不确定性估计生成伪标签
        
        Args:
            images: 输入图像 [B, C, H, W]
            use_teacher: 是否使用教师模型
            
        Returns:
            pseudo_labels: 伪标签 [B, H, W]
            confidence_mask: 置信度掩码 [B, H, W]
            uncertainty: 不确定性图 [B, H, W]
        """
        model_to_use = self.teacher_model if (use_teacher and self.teacher_model is not None) else self.model
        
        # 使用不确定性感知伪标签生成
        pseudo_labels, confidence_mask, uncertainty = self.uncertainty_pseudo_labeling.generate_pseudo_labels(
            model_to_use,
            images
        )
        
        return pseudo_labels, confidence_mask, uncertainty
    
    def _training_step_semi_supervised_with_uncertainty(self, batch, batch_idx):
        """半监督训练步骤（集成不确定性估计）"""
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
        current_contrastive_weight = self.contrastive_weight * ramp_up_factor if self.use_contrastive else 0.0
        
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
        
        # ==================== 2. 对比学习损失 ====================
        if self.use_contrastive and unlabeled_data is not None and labeled_data is not None:
            x_labeled = labeled_data['image']
            y_labeled = labeled_data['mask_del']
            
            # 无标注数据
            if 'image_weak' in unlabeled_data and 'image_strong' in unlabeled_data:
                x_unlabeled_weak = unlabeled_data['image_weak']
                x_unlabeled_strong = unlabeled_data['image_strong']
            else:
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
        
        # ==================== 3. 伪标签损失（使用不确定性估计）⭐ ====================
        if unlabeled_data is not None and pseudo_weight > 0:
            # 使用弱增强图像生成伪标签
            if 'image_weak' in unlabeled_data:
                x_unlabeled_weak = unlabeled_data['image_weak']
            else:
                x_unlabeled_weak = unlabeled_data['image']
            
            # 生成伪标签（使用不确定性估计）
            if self.use_uncertainty:
                pseudo_labels, confidence_mask, uncertainty = self._generate_pseudo_labels_with_uncertainty(
                    x_unlabeled_weak,
                    use_teacher=self.use_ema_teacher
                )
                
                # 记录不确定性统计
                losses_dict['mean_uncertainty'] = uncertainty.mean().item()
                losses_dict['max_uncertainty'] = uncertainty.max().item()
            else:
                # 使用标准方法
                pseudo_labels, confidence_mask = self._generate_pseudo_labels(
                    x_unlabeled_weak,
                    use_teacher=self.use_ema_teacher
                )
                uncertainty = None
            
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
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - 增强版，集成不确定性估计"""
        # 检查批次格式
        if isinstance(batch, dict) and 'labeled' in batch and 'unlabeled' in batch:
            # 半监督数据格式
            return self._training_step_semi_supervised_with_uncertainty(batch, batch_idx)
        else:
            # 标准监督学习格式
            return super(SemiSupervisedContrastiveModule, self).training_step(batch, batch_idx)
    
    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        # 调用父类方法
        super().on_train_epoch_end()
        
        # 记录不确定性统计
        if self.use_uncertainty:
            stats = self.uncertainty_pseudo_labeling.get_uncertainty_stats()
            if stats:
                for key, value in stats.items():
                    if value is not None:
                        self.log(f'uncertainty_{key}', value)
                
                log.info(f"不确定性统计: {stats}")
            
            # 重置统计
            self.uncertainty_pseudo_labeling.reset_stats()


__all__ = ['SemiSupervisedUncertaintyModule']
