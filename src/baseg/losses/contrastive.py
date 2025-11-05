"""
对比学习损失函数
用于半监督学习中的特征表示学习
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log


class SupConLoss(nn.Module):
    """监督对比学习损失 (Supervised Contrastive Learning Loss)
    
    参考论文: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    适用于半监督学习场景，可以处理有标注和无标注数据
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07
    ):
        """
        Args:
            temperature: 温度参数，控制分布的平滑程度
            contrast_mode: 对比模式 ('all' 或 'one')
            base_temperature: 基础温度参数
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            features: [batch_size, n_views, feature_dim] 或 [batch_size, feature_dim]
                     特征向量，已经过projection head
            labels: [batch_size] 标签（可选，用于监督对比学习）
            mask: [batch_size, batch_size] 正样本掩码（可选）
            
        Returns:
            loss: 标量，对比学习损失
        """
        device = features.device
        
        # 处理特征维度
        if len(features.shape) < 3:
            features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # 归一化特征
        features = F.normalize(features, dim=2)
        
        # 展平特征: [batch_size * n_views, feature_dim]
        features = features.view(batch_size * n_views, -1)
        
        # 构建标签掩码
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 无监督对比学习：同一样本的不同view为正样本对
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(n_views, n_views)
            # 移除对角线（自己与自己）
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * n_views).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
        elif labels is not None:
            # 监督对比学习：相同标签的样本为正样本对
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            labels = labels.repeat(n_views, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            # 移除对角线
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * n_views).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
        else:
            mask = mask.float().to(device)
        
        # 计算相似度矩阵
        # [batch_size * n_views, batch_size * n_views]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 为数值稳定性减去最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 构建掩码排除自身
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views).to(device)
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 计算每个正样本对的平均log概率
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # 避免除以0
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class PixelContrastLoss(nn.Module):
    """像素级对比学习损失
    
    适用于密集预测任务（如语义分割）的对比学习
    在特征图的像素级别进行对比
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        ignore_index: int = 255,
        max_samples: int = 1024,  # 每个batch最多采样的像素数
        min_samples_per_class: int = 10  # 每个类别最少采样的像素数
    ):
        """
        Args:
            temperature: 温度参数
            ignore_index: 要忽略的标签索引
            max_samples: 最大采样像素数（控制内存使用）
            min_samples_per_class: 每个类别最小采样数
        """
        super(PixelContrastLoss, self).__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.min_samples_per_class = min_samples_per_class
        
    def _sample_pixels(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从特征图中采样像素
        
        Args:
            features: [B, C, H, W] 特征图
            labels: [B, H, W] 标签
            
        Returns:
            sampled_features: [N, C] 采样的特征
            sampled_labels: [N] 采样的标签
        """
        B, C, H, W = features.shape
        
        # 展平
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels = labels.reshape(-1)  # [B*H*W]
        
        # 移除ignore_index
        valid_mask = labels != self.ignore_index
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 如果样本数太多，进行采样
        if features.shape[0] > self.max_samples:
            # 分层采样，确保每个类别都有代表
            unique_labels = torch.unique(labels)
            sampled_indices = []
            
            samples_per_class = self.max_samples // len(unique_labels)
            samples_per_class = max(samples_per_class, self.min_samples_per_class)
            
            for label in unique_labels:
                label_mask = labels == label
                label_indices = torch.where(label_mask)[0]
                
                if len(label_indices) > samples_per_class:
                    # 随机采样
                    perm = torch.randperm(len(label_indices))[:samples_per_class]
                    sampled_indices.append(label_indices[perm])
                else:
                    sampled_indices.append(label_indices)
            
            sampled_indices = torch.cat(sampled_indices)
            features = features[sampled_indices]
            labels = labels[sampled_indices]
        
        return features, labels
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] 特征图
            labels: [B, H, W] 标签图
            
        Returns:
            loss: 标量，像素级对比损失
        """
        # 采样像素
        sampled_features, sampled_labels = self._sample_pixels(features, labels)
        
        if sampled_features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 归一化特征
        sampled_features = F.normalize(sampled_features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(
            sampled_features, 
            sampled_features.T
        ) / self.temperature
        
        # 构建正样本掩码（相同标签的像素）
        labels_matrix = sampled_labels.unsqueeze(0) == sampled_labels.unsqueeze(1)
        labels_matrix = labels_matrix.float()
        
        # 移除对角线（自己与自己）
        logits_mask = 1 - torch.eye(sampled_features.shape[0]).to(features.device)
        labels_matrix = labels_matrix * logits_mask
        
        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 计算每个样本的平均正样本log概率
        pos_mask_sum = labels_matrix.sum(1)
        pos_mask_sum = torch.clamp(pos_mask_sum, min=1.0)
        mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / pos_mask_sum
        
        # 最终损失
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ProjectionHead(nn.Module):
    """投影头，用于对比学习
    
    将backbone输出的特征投影到对比学习空间
    """
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int = 256, 
        out_dim: int = 128,
        num_layers: int = 2,
        use_bn: bool = True
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出特征维度
            num_layers: MLP层数
            use_bn: 是否使用BatchNorm
        """
        super(ProjectionHead, self).__init__()
        
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        
        # 最后一层不使用激活函数
        layers.append(nn.Linear(current_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim] 或 [B, in_dim, H, W]
            
        Returns:
            [B, out_dim] 投影后的特征
        """
        # 如果是4D特征图，先进行全局平均池化
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
        
        return self.mlp(x)


# 导出列表
__all__ = ['SupConLoss', 'PixelContrastLoss', 'ProjectionHead']

