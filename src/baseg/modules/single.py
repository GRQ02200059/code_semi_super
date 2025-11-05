from typing import Any, Callable
from loguru import logger as log
import torch
import torchvision
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss
from baseg.modules.base import BaseModule


class SingleTaskModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
    ):
        super().__init__(config, tiler, predict_callback)
        self.loss_type = loss
        if loss == "bce":
            self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255, pos_weight=torch.tensor(3.0))
        else:
            self.criterion_decode = DiceLoss(mode="binary", from_logits=True, ignore_index=255)
        
        # 记录初始化信息
        log.info(f"Initialized SingleTaskModule with loss: {loss}")
        log.info(f"Model config: {config['type']}")
        
        # 设置可视化频率
        self.viz_interval = 50  # 每50个batch可视化一次

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]

        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        # 尺寸不一致时插值
        if decode_out.shape[-2:] != y_del.shape[-2:]:
            import torch.nn.functional as F
            decode_out = F.interpolate(decode_out, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        # 记录每个batch的详细信息
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_loss_decode", loss_decode, on_step=True)
        
        # 记录当前学习率
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True)
        
        # 计算并记录每个batch的指标
        for metric_name, metric in self.train_metrics.items():
            metric_value = metric(decode_out.squeeze(1), y_del.float())
            self.log(f"{metric_name}_step", metric_value, on_step=True)
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        
        # 记录正样本比例
        pos_ratio = torch.mean((y_del > 0).float())
        self.log("train_pos_ratio", pos_ratio, on_step=True)
        
        
        # 每100个batch打印一次详细信息
        if batch_idx % 100 == 0:
            log.info(f"Batch {batch_idx}: loss={loss:.4f}, lr={current_lr:.6f}")
        
        # 定期可视化训练结果
        if batch_idx % self.viz_interval == 0:
            self._log_images(x, y_del, decode_out, "train", batch_idx)
            
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        # 尺寸不一致时插值
        if decode_out.shape[-2:] != y_del.shape[-2:]:
            import torch.nn.functional as F
            decode_out = F.interpolate(decode_out, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        # 记录验证指标
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss_decode", loss_decode, on_step=True, on_epoch=True)
        
        # 计算并记录每个batch的指标
        batch_metrics = {}
        for metric_name, metric in self.val_metrics.items():
            metric_value = metric(decode_out.squeeze(1), y_del.float())
            batch_metrics[metric_name] = metric_value.item()
            self.log(f"{metric_name}_step", metric_value, on_step=True)
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        
        # 记录正样本比例
        pos_ratio = torch.mean((y_del > 0).float())
        self.log("val_pos_ratio", pos_ratio, on_step=True, on_epoch=True)
        
        
        # 每10个batch打印一次验证信息
        if batch_idx % 10 == 0:
            log.info(f"Val Batch {batch_idx}: loss={loss:.4f}, metrics={batch_metrics}")
        
        # 可视化验证结果
        if batch_idx == 0:  # 每个epoch只可视化第一个batch
            self._log_images(x, y_del, decode_out, "val", self.current_epoch)
            
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        # 尺寸不一致时插值
        if decode_out.shape[-2:] != y_del.shape[-2:]:
            import torch.nn.functional as F
            decode_out = F.interpolate(decode_out, size=y_del.shape[-2:], mode='bilinear', align_corners=False)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        # 记录测试指标
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_loss_decode", loss_decode, on_epoch=True, logger=True)
        
        # 计算并记录详细指标
        batch_metrics = {}
        for metric_name, metric in self.test_metrics.items():
            metric_value = metric(decode_out.squeeze(1), y_del.float())
            batch_metrics[metric_name] = metric_value.item()
            self.log(metric_name, metric, on_epoch=True, logger=True)
        
        # 记录每个batch的测试信息
        log.info(f"Test Batch {batch_idx}: loss={loss:.4f}, metrics={batch_metrics}")
        
        # 可视化测试结果
        if batch_idx < 5:  # 只可视化前5个batch
            self._log_images(x, y_del, decode_out, "test", batch_idx)
        
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch["S2L2A"]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

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
            patience=3, 
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
        # 记录每个epoch结束时的指标
        metrics_str = ", ".join([f"{name}={metric.compute():.4f}" for name, metric in self.train_metrics.items()])
        log.info(f"Epoch {self.current_epoch} ended. Metrics: {metrics_str}")
        
    def on_validation_epoch_end(self):
        # 记录每个验证epoch结束时的指标
        metrics_str = ", ".join([f"{name}={metric.compute():.4f}" for name, metric in self.val_metrics.items()])
        log.info(f"Validation Epoch {self.current_epoch} ended. Metrics: {metrics_str}")
        
    def _log_images(self, images, masks, predictions, stage, step_or_epoch):
        """将图像、真实标签和预测结果记录到TensorBoard"""
        # 选择最多4张图像进行可视化
        n = min(4, images.size(0))
        
        # 准备RGB图像（仅使用前3个波段）
        rgb_images = images[:n, :3]  # 只取前3个波段作为RGB
        
        # 归一化到[0,1]范围以便可视化
        for i in range(n):
            for c in range(3):
                channel = rgb_images[i, c]
                if channel.max() > channel.min():
                    rgb_images[i, c] = (channel - channel.min()) / (channel.max() - channel.min())
        
        # 准备标签和预测
        true_masks = masks[:n].unsqueeze(1)  # [N, 1, H, W]
        pred_masks = torch.sigmoid(predictions[:n])  # [N, 1, H, W]
        pred_binary = (pred_masks > 0.5).float()  # 二值化预测
        
        # 创建可视化网格
        vis_images = torch.cat([
            rgb_images,  # 原始RGB图像
            true_masks.repeat(1, 3, 1, 1),  # 真实标签（复制到3通道）
            pred_binary.repeat(1, 3, 1, 1),  # 预测结果（复制到3通道）
        ], dim=0)
        
        # 创建图像网格
        grid = torchvision.utils.make_grid(vis_images, nrow=n, normalize=False)
        
        # 记录到TensorBoard
        self.logger.experiment.add_image(f'{stage}_predictions', grid, global_step=step_or_epoch)
        
        # 记录预测概率直方图
        self.logger.experiment.add_histogram(f'{stage}_pred_probs', pred_masks, global_step=step_or_epoch)
