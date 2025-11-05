from typing import Any, Callable, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger as log

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss
from baseg.modules.base import BaseModule


class SemiSupervisedModule(BaseModule):
    """半监督学习模块，支持伪标签生成和一致性正则化"""
    
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
        # 半监督学习参数
        pseudo_threshold: float = 0.95,  # 伪标签置信度阈值
        consistency_weight: float = 1.0,  # 一致性损失权重
        pseudo_weight: float = 1.0,      # 伪标签损失权重
        ramp_up_epochs: int = 10,        # 权重渐进增加的轮数
        ema_decay: float = 0.99,         # 教师模型EMA衰减率
        use_ema_teacher: bool = True,    # 是否使用EMA教师模型
    ):
        super().__init__(config, tiler, predict_callback)
        
        # 损失函数
        self.loss_type = loss
        if loss == "bce":
            self.criterion_supervised = SoftBCEWithLogitsLoss(ignore_index=255, pos_weight=torch.tensor(3.0))
            self.criterion_unsupervised = SoftBCEWithLogitsLoss(ignore_index=255, pos_weight=torch.tensor(1.0))
        else:
            self.criterion_supervised = DiceLoss(mode="binary", from_logits=True, ignore_index=255)
            self.criterion_unsupervised = DiceLoss(mode="binary", from_logits=True, ignore_index=255)
        
        # 一致性损失（MSE）
        self.consistency_criterion = torch.nn.MSELoss()
        
        # 半监督学习参数
        self.pseudo_threshold = pseudo_threshold
        self.consistency_weight = consistency_weight
        self.pseudo_weight = pseudo_weight
        self.ramp_up_epochs = ramp_up_epochs
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        
        # EMA教师模型
        if self.use_ema_teacher:
            import copy
            # 立即创建教师模型，确保checkpoint可以正确加载
            self.teacher_model = copy.deepcopy(self.model)
            # 教师模型设为评估模式
            self.teacher_model.eval()
            # 教师模型参数不需要梯度
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            log.info("EMA教师模型初始化完成")
        
        # 记录统计信息
        self.pseudo_label_stats = {
            'total_pixels': 0,
            'pseudo_pixels': 0,
            'positive_pseudo': 0
        }
        
        log.info(f"初始化半监督学习模块: threshold={pseudo_threshold}, "
                f"consistency_weight={consistency_weight}, pseudo_weight={pseudo_weight}")

    def _init_teacher_model(self):
        """初始化EMA教师模型"""
        if self.teacher_model is None and self.use_ema_teacher:
            import copy
            # 使用深拷贝创建教师模型
            self.teacher_model = copy.deepcopy(self.model)
            # 教师模型设为评估模式
            self.teacher_model.eval()
            # 教师模型参数不需要梯度
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            log.info("EMA教师模型初始化完成")

    def _update_teacher_model(self):
        """使用EMA更新教师模型参数"""
        if not self.use_ema_teacher or self.teacher_model is None:
            return
            
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                self.model.parameters()
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data + 
                    (1 - self.ema_decay) * student_param.data
                )

    def _get_current_weights(self, epoch: int) -> Tuple[float, float]:
        """获取当前轮次的权重（渐进式增加）"""
        if epoch < self.ramp_up_epochs:
            # 线性增加权重
            ramp_up_factor = epoch / self.ramp_up_epochs
        else:
            ramp_up_factor = 1.0
            
        current_consistency_weight = self.consistency_weight * ramp_up_factor
        current_pseudo_weight = self.pseudo_weight * ramp_up_factor
        
        return current_consistency_weight, current_pseudo_weight

    def _generate_pseudo_labels(self, images: torch.Tensor, use_teacher: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成伪标签
        
        Args:
            images: 输入图像 [B, C, H, W]
            use_teacher: 是否使用教师模型生成伪标签
            
        Returns:
            pseudo_labels: 伪标签 [B, H, W]
            confidence_mask: 置信度掩膜 [B, H, W]
        """
        model_to_use = self.teacher_model if (use_teacher and self.teacher_model is not None) else self.model
        
        with torch.no_grad():
            model_to_use.eval()
            logits = model_to_use(images)
            if logits.dim() == 4:
                logits = logits.squeeze(1)  # [B, H, W]
            
            # 计算概率
            probs = torch.sigmoid(logits)
            
            # 生成伪标签（二值化）
            pseudo_labels = (probs > 0.5).float()
            
            # 计算置信度掩膜
            confidence = torch.max(probs, 1 - probs)  # 取max(p, 1-p)
            confidence_mask = (confidence > self.pseudo_threshold).float()
            
            return pseudo_labels, confidence_mask

    def _compute_consistency_loss(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """计算一致性损失"""
        # 将logits转换为概率
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)
        
        # 计算MSE损失
        consistency_loss = self.consistency_criterion(prob1, prob2)
        return consistency_loss

    def training_step(self, batch: Any, batch_idx: int):
        # 初始化教师模型
        if self.teacher_model is None and self.use_ema_teacher:
            self._init_teacher_model()
        
        # 获取当前权重
        current_consistency_weight, current_pseudo_weight = self._get_current_weights(self.current_epoch)
        
        # 处理半监督批次
        if isinstance(batch, dict) and 'labeled' in batch and 'unlabeled' in batch:
            labeled_data = batch['labeled']
            unlabeled_data = batch['unlabeled']
            
            # 检查数据是否为空
            if labeled_data is None and unlabeled_data is None:
                log.warning("批次数据为空，跳过该步骤")
                return None
            
            return self._training_step_mixed(
                labeled_data, unlabeled_data, batch_idx, 
                current_consistency_weight, current_pseudo_weight
            )
        else:
            # 传统的有标注数据训练
            return self._training_step_supervised(batch, batch_idx)

    def _training_step_mixed(self, labeled_data: Dict, unlabeled_data: Dict, batch_idx: int, 
                           consistency_weight: float, pseudo_weight: float):
        """混合训练步骤（有标注 + 无标注数据）"""
        total_loss = 0.0
        
        # === 有标注数据的监督学习 ===
        x_labeled = labeled_data["S2L2A"]
        y_labeled = labeled_data["DEL"]
        
        pred_labeled = self.model(x_labeled)
        if pred_labeled.shape[-2:] != y_labeled.shape[-2:]:
            pred_labeled = F.interpolate(pred_labeled, size=y_labeled.shape[-2:], 
                                       mode='bilinear', align_corners=False)
        
        supervised_loss = self.criterion_supervised(pred_labeled.squeeze(1), y_labeled.float())
        total_loss += supervised_loss
        
        # === 无标注数据的半监督学习 ===
        if unlabeled_data is not None and len(unlabeled_data["S2L2A"]) > 0:
            x_unlabeled = unlabeled_data["S2L2A"]
            
            # 1. 生成伪标签
            pseudo_labels, confidence_mask = self._generate_pseudo_labels(x_unlabeled, use_teacher=True)
            
            # 2. 学生模型预测
            pred_unlabeled = self.model(x_unlabeled)
            if pred_unlabeled.shape[-2:] != pseudo_labels.shape[-2:]:
                pred_unlabeled = F.interpolate(pred_unlabeled, size=pseudo_labels.shape[-2:], 
                                             mode='bilinear', align_corners=False)
            
            # 3. 伪标签损失（只在高置信度区域计算）
            if confidence_mask.sum() > 0:
                # 创建有效掩膜
                valid_mask = confidence_mask
                
                # 计算伪标签损失 - 修复损失计算方式
                pred_for_pseudo = pred_unlabeled.squeeze(1)
                target_for_pseudo = pseudo_labels
                
                # 创建掩码版本的预测和目标
                masked_pred = pred_for_pseudo * valid_mask
                masked_target = target_for_pseudo * valid_mask
                
                # 只在有效区域计算损失
                if hasattr(self.criterion_unsupervised, 'ignore_index'):
                    # 对于支持ignore_index的损失函数，将无效区域设为ignore值
                    final_target = target_for_pseudo.clone()
                    final_target[valid_mask == 0] = 255  # ignore_index
                    pseudo_loss = self.criterion_unsupervised(pred_for_pseudo, final_target)
                else:
                    # 对于不支持ignore_index的损失函数，手动计算平均
                    pixel_losses = F.binary_cross_entropy_with_logits(
                        pred_for_pseudo, target_for_pseudo, reduction='none'
                    )
                    masked_losses = pixel_losses * valid_mask
                    pseudo_loss = masked_losses.sum() / (valid_mask.sum() + 1e-8)
                
                total_loss += pseudo_weight * pseudo_loss
                
                # 更新统计信息
                self.pseudo_label_stats['total_pixels'] += valid_mask.numel()
                self.pseudo_label_stats['pseudo_pixels'] += valid_mask.sum().item()
                self.pseudo_label_stats['positive_pseudo'] += (pseudo_labels * valid_mask).sum().item()
                
                # 记录伪标签损失
                self.log("pseudo_loss", pseudo_loss, on_step=True, prog_bar=True)
            
            # 4. 一致性正则化（如果有多个增强版本）
            if 'S2L2A_aug' in unlabeled_data:
                x_unlabeled_aug = unlabeled_data["S2L2A_aug"]
                pred_unlabeled_aug = self.model(x_unlabeled_aug)
                if pred_unlabeled_aug.shape[-2:] != pred_unlabeled.shape[-2:]:
                    pred_unlabeled_aug = F.interpolate(pred_unlabeled_aug, size=pred_unlabeled.shape[-2:], 
                                                     mode='bilinear', align_corners=False)
                
                consistency_loss = self._compute_consistency_loss(
                    pred_unlabeled.squeeze(1), 
                    pred_unlabeled_aug.squeeze(1)
                )
                total_loss += consistency_weight * consistency_loss
                
                self.log("consistency_loss", consistency_loss, on_step=True, prog_bar=True)
        
        # 更新教师模型
        self._update_teacher_model()
        
        # 记录损失
        self.log("train_loss", total_loss, on_step=True, prog_bar=True)
        self.log("supervised_loss", supervised_loss, on_step=True)
        self.log("consistency_weight", consistency_weight, on_step=True)
        self.log("pseudo_weight", pseudo_weight, on_step=True)
        
        # 计算并记录指标
        for metric_name, metric in self.train_metrics.items():
            metric_value = metric(pred_labeled.squeeze(1), y_labeled.float())
            self.log(f"{metric_name}_step", metric_value, on_step=True)
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        
        # 记录伪标签统计
        if batch_idx % 100 == 0 and self.pseudo_label_stats['total_pixels'] > 0:
            pseudo_ratio = self.pseudo_label_stats['pseudo_pixels'] / self.pseudo_label_stats['total_pixels']
            positive_ratio = self.pseudo_label_stats['positive_pseudo'] / max(self.pseudo_label_stats['pseudo_pixels'], 1)
            log.info(f"Batch {batch_idx}: pseudo_ratio={pseudo_ratio:.3f}, positive_ratio={positive_ratio:.3f}")
            
            self.log("pseudo_label_ratio", pseudo_ratio, on_step=True)
            self.log("pseudo_positive_ratio", positive_ratio, on_step=True)
        
        return total_loss

    def _training_step_supervised(self, batch: Any, batch_idx: int):
        """传统的监督学习训练步骤"""
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        
        pred = self.model(x)
        if pred.shape[-2:] != y_del.shape[-2:]:
            pred = F.interpolate(pred, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = self.criterion_supervised(pred.squeeze(1), y_del.float())
        
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        
        # 计算并记录指标
        for metric_name, metric in self.train_metrics.items():
            metric_value = metric(pred.squeeze(1), y_del.float())
            self.log(f"{metric_name}_step", metric_value, on_step=True)
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """验证步骤（与原来相同）"""
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        
        pred = self.model(x)
        if pred.shape[-2:] != y_del.shape[-2:]:
            pred = F.interpolate(pred, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = self.criterion_supervised(pred.squeeze(1), y_del.float())
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # 计算并记录指标
        for metric_name, metric in self.val_metrics.items():
            metric_value = metric(pred.squeeze(1), y_del.float())
            self.log(f"{metric_name}_step", metric_value, on_step=True)
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        """测试步骤（与原来相同）"""
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        
        pred = self.model(x)
        if pred.shape[-2:] != y_del.shape[-2:]:
            pred = F.interpolate(pred, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = self.criterion_supervised(pred.squeeze(1), y_del.float())
        
        self.log("test_loss", loss, on_epoch=True, logger=True)
        
        # 计算并记录指标
        for metric_name, metric in self.test_metrics.items():
            metric_value = metric(pred.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, logger=True)
        
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """预测步骤"""
        full_image = batch["S2L2A"]

        def callback(batch: Any):
            pred = self.model(batch)
            return pred.squeeze(1) if pred.dim() == 4 else pred

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,  # 半监督学习通常需要更多耐心
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
    def on_train_epoch_end(self):
        """训练轮次结束时的处理"""
        # 重置伪标签统计
        self.pseudo_label_stats = {
            'total_pixels': 0,
            'pseudo_pixels': 0,
            'positive_pseudo': 0
        }
        
        # 记录指标
        metrics_str = ", ".join([f"{name}={metric.compute():.4f}" for name, metric in self.train_metrics.items()])
        log.info(f"Semi-supervised Epoch {self.current_epoch} ended. Metrics: {metrics_str}")
