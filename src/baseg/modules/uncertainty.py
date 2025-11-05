"""
不确定性估计模块
用于半监督学习中的动态伪标签阈值调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger as log


class UncertaintyEstimator:
    """基于Monte Carlo Dropout的不确定性估计
    
    通过多次前向传播（启用dropout）来估计模型的预测不确定性
    """
    
    def __init__(
        self,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
        min_threshold: float = 0.8,
        max_threshold: float = 0.99,
        uncertainty_weight: float = 0.1
    ):
        """
        Args:
            n_samples: Monte Carlo采样次数
            dropout_rate: Dropout比率（如果模型支持动态设置）
            min_threshold: 最小阈值
            max_threshold: 最大阈值
            uncertainty_weight: 不确定性对阈值的影响权重
        """
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.uncertainty_weight = uncertainty_weight
        
        log.info(f"初始化不确定性估计器: n_samples={n_samples}, "
                f"uncertainty_weight={uncertainty_weight}")
    
    def enable_dropout(self, model: nn.Module) -> None:
        """启用模型中的所有Dropout层"""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    def estimate_uncertainty(
        self, 
        model: nn.Module, 
        x: torch.Tensor,
        return_all_predictions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        估计预测的不确定性
        
        Args:
            model: 神经网络模型
            x: 输入图像 [B, C, H, W]
            return_all_predictions: 是否返回所有预测
            
        Returns:
            mean_pred: 平均预测 [B, 1, H, W]
            uncertainty: 预测方差（不确定性）[B, 1, H, W]
            (predictions): 可选，所有预测 [n_samples, B, 1, H, W]
        """
        # 保存原始训练状态
        was_training = model.training
        
        # 设置为评估模式，但保持dropout启用
        model.eval()
        self.enable_dropout(model)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(self.n_samples):
                # 前向传播
                pred = model(x)
                
                # 确保形状正确
                if pred.dim() == 4:
                    pred = pred.squeeze(1) if pred.shape[1] == 1 else pred
                
                # 转换为概率
                pred = torch.sigmoid(pred)
                
                # 添加通道维度如果需要
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                
                predictions.append(pred)
        
        # 恢复原始训练状态
        if was_training:
            model.train()
        else:
            model.eval()
        
        # 堆叠所有预测 [n_samples, B, 1, H, W]
        predictions = torch.stack(predictions, dim=0)
        
        # 计算均值和方差
        mean_pred = predictions.mean(dim=0)  # [B, 1, H, W]
        uncertainty = predictions.var(dim=0)  # [B, 1, H, W]
        
        if return_all_predictions:
            return mean_pred, uncertainty, predictions
        else:
            return mean_pred, uncertainty
    
    def adaptive_threshold(
        self, 
        uncertainty: torch.Tensor, 
        base_threshold: float = 0.9
    ) -> torch.Tensor:
        """根据不确定性动态调整阈值
        
        不确定性越高，阈值越高（更保守）
        
        Args:
            uncertainty: 预测不确定性 [B, 1, H, W]
            base_threshold: 基础阈值
            
        Returns:
            adaptive_threshold: 自适应阈值 [B, 1, H, W]
        """
        # 归一化不确定性到[0, 1]
        uncertainty_normalized = uncertainty / (uncertainty.max() + 1e-8)
        
        # 根据不确定性调整阈值
        # 不确定性越高，阈值越高
        adaptive_t = base_threshold + uncertainty_normalized * self.uncertainty_weight
        
        # 限制在合理范围内
        adaptive_t = adaptive_t.clamp(self.min_threshold, self.max_threshold)
        
        return adaptive_t
    
    def compute_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算预测熵作为不确定性度量
        
        Args:
            predictions: 预测概率 [B, 1, H, W]
            
        Returns:
            entropy: 熵值 [B, 1, H, W]
        """
        # 避免log(0)
        predictions = predictions.clamp(1e-8, 1 - 1e-8)
        
        # 计算二元熵
        entropy = -(predictions * torch.log(predictions) + 
                   (1 - predictions) * torch.log(1 - predictions))
        
        return entropy
    
    def mutual_information(
        self, 
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """计算互信息作为认知不确定性度量
        
        Args:
            predictions: 所有预测 [n_samples, B, 1, H, W]
            
        Returns:
            mi: 互信息 [B, 1, H, W]
        """
        # 平均预测
        mean_pred = predictions.mean(dim=0)
        
        # 平均预测的熵（总不确定性）
        total_entropy = self.compute_entropy(mean_pred)
        
        # 每个预测的熵的平均（偶然不确定性）
        entropies = torch.stack([self.compute_entropy(p) for p in predictions])
        aleatoric_uncertainty = entropies.mean(dim=0)
        
        # 互信息（认知不确定性）= 总不确定性 - 偶然不确定性
        mi = total_entropy - aleatoric_uncertainty
        
        return mi


class EnsembleUncertaintyEstimator:
    """基于集成的不确定性估计
    
    使用多个模型（如EMA教师模型 + 学生模型）进行集成预测
    """
    
    def __init__(
        self,
        min_threshold: float = 0.8,
        max_threshold: float = 0.99,
        uncertainty_weight: float = 0.1
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.uncertainty_weight = uncertainty_weight
    
    def estimate_uncertainty(
        self,
        models: list,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用多个模型估计不确定性
        
        Args:
            models: 模型列表（如[student, teacher]）
            x: 输入图像
            
        Returns:
            mean_pred: 平均预测
            uncertainty: 预测方差
        """
        predictions = []
        
        with torch.no_grad():
            for model in models:
                was_training = model.training
                model.eval()
                
                pred = model(x)
                if pred.dim() == 4 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                pred = torch.sigmoid(pred)
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                
                predictions.append(pred)
                
                if was_training:
                    model.train()
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty
    
    def adaptive_threshold(
        self,
        uncertainty: torch.Tensor,
        base_threshold: float = 0.9
    ) -> torch.Tensor:
        """动态阈值调整"""
        uncertainty_normalized = uncertainty / (uncertainty.max() + 1e-8)
        adaptive_t = base_threshold + uncertainty_normalized * self.uncertainty_weight
        return adaptive_t.clamp(self.min_threshold, self.max_threshold)


class UncertaintyAwarePseudoLabeling:
    """不确定性感知的伪标签生成"""
    
    def __init__(
        self,
        estimator: UncertaintyEstimator,
        base_threshold: float = 0.9,
        use_adaptive_threshold: bool = True,
        log_uncertainty: bool = True
    ):
        """
        Args:
            estimator: 不确定性估计器
            base_threshold: 基础伪标签阈值
            use_adaptive_threshold: 是否使用自适应阈值
            log_uncertainty: 是否记录不确定性统计
        """
        self.estimator = estimator
        self.base_threshold = base_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.log_uncertainty = log_uncertainty
        
        # 统计信息
        self.uncertainty_stats = {
            'mean_uncertainty': [],
            'max_uncertainty': [],
            'adaptive_threshold_mean': []
        }
    
    def generate_pseudo_labels(
        self,
        model: nn.Module,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        生成不确定性感知的伪标签
        
        Args:
            model: 教师模型
            images: 输入图像 [B, C, H, W]
            
        Returns:
            pseudo_labels: 伪标签 [B, H, W]
            confidence_mask: 置信度掩码 [B, H, W]
            uncertainty: 不确定性图（可选）[B, 1, H, W]
        """
        # 估计不确定性
        mean_pred, uncertainty = self.estimator.estimate_uncertainty(model, images)
        
        # 去除通道维度
        if mean_pred.dim() == 4:
            mean_pred = mean_pred.squeeze(1)
        if uncertainty.dim() == 4:
            uncertainty_map = uncertainty.squeeze(1)
        else:
            uncertainty_map = uncertainty
        
        # 生成伪标签
        pseudo_labels = (mean_pred > 0.5).float()
        
        # 计算置信度
        confidence = torch.max(mean_pred, 1 - mean_pred)
        
        # 自适应阈值或固定阈值
        if self.use_adaptive_threshold:
            # 添加通道维度以匹配
            uncertainty_for_threshold = uncertainty if uncertainty.dim() == 4 else uncertainty.unsqueeze(1)
            adaptive_t = self.estimator.adaptive_threshold(
                uncertainty_for_threshold, 
                self.base_threshold
            )
            if adaptive_t.dim() == 4:
                adaptive_t = adaptive_t.squeeze(1)
            
            confidence_mask = (confidence > adaptive_t).float()
            
            # 记录统计
            if self.log_uncertainty:
                self.uncertainty_stats['adaptive_threshold_mean'].append(
                    adaptive_t.mean().item()
                )
        else:
            confidence_mask = (confidence > self.base_threshold).float()
        
        # 记录不确定性统计
        if self.log_uncertainty:
            self.uncertainty_stats['mean_uncertainty'].append(
                uncertainty_map.mean().item()
            )
            self.uncertainty_stats['max_uncertainty'].append(
                uncertainty_map.max().item()
            )
        
        # 返回不确定性图用于可视化
        return pseudo_labels, confidence_mask, uncertainty_map
    
    def get_uncertainty_stats(self) -> dict:
        """获取不确定性统计信息"""
        if not self.uncertainty_stats['mean_uncertainty']:
            return {}
        
        import numpy as np
        return {
            'mean_uncertainty': np.mean(self.uncertainty_stats['mean_uncertainty']),
            'max_uncertainty': np.max(self.uncertainty_stats['max_uncertainty']),
            'adaptive_threshold_mean': np.mean(self.uncertainty_stats['adaptive_threshold_mean']) 
                if self.uncertainty_stats['adaptive_threshold_mean'] else None
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.uncertainty_stats = {
            'mean_uncertainty': [],
            'max_uncertainty': [],
            'adaptive_threshold_mean': []
        }


__all__ = [
    'UncertaintyEstimator',
    'EnsembleUncertaintyEstimator', 
    'UncertaintyAwarePseudoLabeling'
]

