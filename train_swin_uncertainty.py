#!/usr/bin/env python3
"""
Swin Transformer + 半监督学习 + 对比学习 + 不确定性估计 训练脚本

集成五重学习策略，通过不确定性估计动态调整伪标签阈值
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from loguru import logger as log

# 添加src到路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from baseg.datamodules_semi import SemiSupervisedEMSDataModule
from baseg.modules.semi_supervised_uncertainty import SemiSupervisedUncertaintyModule


def set_deterministic_training(seed=42):
    """设置确定性训练环境"""
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 设置PyTorch Lightning的随机种子
    seed_everything(seed, workers=True)
    
    log.info(f"✅ 确定性训练环境设置完成 (seed={seed})")


def main():
    """主训练函数"""
    # 设置确定性训练
    set_deterministic_training(42)
    
    # 配置文件路径
    config_path = "configs/semi_supervised/ems_swin_semi_uncertainty.py"
    
    # 加载配置
    log.info(f"📂 加载配置文件: {config_path}")
    config = Config.fromfile(config_path)
    
    # 设置实验名称（添加时间戳）
    labeled_ratio = config['data']['labeled_ratio']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['name']}_labeled{int(labeled_ratio*100)}pct_{timestamp}"
    config["name"] = exp_name
    log.info(f"🏷️  实验名称: {exp_name}")
    log.info(f"⏰  时间戳: {timestamp}")
    
    # 创建数据模块
    log.info("📊 创建Swin Transformer半监督数据模块...")
    datamodule = SemiSupervisedEMSDataModule(**config["data"])
    
    # 创建模型模块（使用不确定性估计增强版）
    log.info("🧠 创建Swin Transformer + 对比学习 + 不确定性估计模块...")
    model_config = config["model"]
    semi_config = config["semi_supervised"]
    loss = config.get("loss", "bce")
    
    # 提取所有参数
    module_params = {
        # 半监督参数
        'pseudo_threshold': semi_config["pseudo_threshold"],
        'consistency_weight': semi_config["consistency_weight"],
        'pseudo_weight': semi_config["pseudo_weight"],
        'ramp_up_epochs': semi_config["ramp_up_epochs"],
        'ema_decay': semi_config["ema_decay"],
        'use_ema_teacher': semi_config["use_ema_teacher"],
        # 对比学习参数
        'use_contrastive': semi_config.get('use_contrastive', True),
        'contrastive_weight': semi_config.get('contrastive_weight', 0.5),
        'contrastive_temperature': semi_config.get('contrastive_temperature', 0.07),
        'contrastive_mode': semi_config.get('contrastive_mode', 'global'),
        'projection_dim': semi_config.get('projection_dim', 128),
        'projection_hidden_dim': semi_config.get('projection_hidden_dim', 256),
        # 不确定性估计参数 ⭐
        'use_uncertainty': semi_config.get('use_uncertainty', True),
        'uncertainty_n_samples': semi_config.get('uncertainty_n_samples', 5),
        'uncertainty_weight': semi_config.get('uncertainty_weight', 0.1),
        'min_threshold': semi_config.get('min_threshold', 0.8),
        'max_threshold': semi_config.get('max_threshold', 0.99),
        'use_adaptive_threshold': semi_config.get('use_adaptive_threshold', True),
    }
    
    module = SemiSupervisedUncertaintyModule(
        model_config,
        loss=loss,
        **module_params
    )
    
    # 初始化预训练权重
    module.init_pretrained()
    log.info("✅ Swin Transformer预训练权重初始化完成")
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="outputs",
        name=exp_name,
        default_hp_metric=False,
        log_graph=True,
    )
    
    # 记录超参数
    hyperparams = {
        "model_type": "SwinTransformer_Contrastive_Uncertainty",
        "backbone": model_config["backbone"]["type"],
        "loss": loss,
        # 数据参数
        "labeled_ratio": labeled_ratio,
        "labeled_batch_size": config["data"]["labeled_batch_size"],
        "unlabeled_batch_size": config["data"]["unlabeled_batch_size"],
        "patch_size": config["data"]["patch_size"],
        # 半监督参数
        "pseudo_threshold": module_params['pseudo_threshold'],
        "consistency_weight": module_params['consistency_weight'],
        "pseudo_weight": module_params['pseudo_weight'],
        "ramp_up_epochs": module_params['ramp_up_epochs'],
        "ema_decay": module_params['ema_decay'],
        # 对比学习参数
        "use_contrastive": module_params['use_contrastive'],
        "contrastive_weight": module_params['contrastive_weight'],
        "contrastive_temperature": module_params['contrastive_temperature'],
        "contrastive_mode": module_params['contrastive_mode'],
        "projection_dim": module_params['projection_dim'],
        # 不确定性估计参数 ⭐
        "use_uncertainty": module_params['use_uncertainty'],
        "uncertainty_n_samples": module_params['uncertainty_n_samples'],
        "uncertainty_weight": module_params['uncertainty_weight'],
        "use_adaptive_threshold": module_params['use_adaptive_threshold'],
        # 训练参数
        "max_epochs": config["trainer"]["max_epochs"],
    }
    logger.log_hyperparams(hyperparams)
    
    # 打印不确定性估计配置
    log.info("🎯 不确定性估计配置:")
    log.info(f"  - 启用: {module_params['use_uncertainty']}")
    if module_params['use_uncertainty']:
        log.info(f"  - 采样次数: {module_params['uncertainty_n_samples']}")
        log.info(f"  - 不确定性权重: {module_params['uncertainty_weight']}")
        log.info(f"  - 自适应阈值: {module_params['use_adaptive_threshold']}")
        log.info(f"  - 阈值范围: [{module_params['min_threshold']}, {module_params['max_threshold']}]")
    
    # 创建回调函数
    callbacks = []
    
    # 早停回调
    if "early_stopping" in config:
        early_stop_callback = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            patience=config["early_stopping"]["patience"],
            mode=config["early_stopping"]["mode"],
            verbose=config["early_stopping"]["verbose"],
        )
        callbacks.append(early_stop_callback)
        log.info(f"📉 早停策略: 监控{config['early_stopping']['monitor']}, "
                f"patience={config['early_stopping']['patience']}")
    
    # 模型检查点回调 - IoU
    checkpoint_callback_iou = ModelCheckpoint(
        dirpath=f"outputs/{exp_name}/version_0/weights/iou",
        filename="best-epoch={epoch:02d}-val_iou={val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=3,
        save_last=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback_iou)
    
    # 模型检查点回调 - Loss
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=f"outputs/{exp_name}/version_0/weights/loss",
        filename="model-epoch={epoch:02d}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback_loss)
    
    # 创建训练器
    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
    )
    
    # 打印训练配置摘要
    log.info("=" * 80)
    log.info("🚀 训练配置摘要")
    log.info("=" * 80)
    log.info(f"实验名称: {exp_name}")
    log.info(f"模型架构: Swin Transformer + SegFormer Head + Contrastive + Uncertainty")
    log.info(f"训练轮数: {config['trainer']['max_epochs']}")
    log.info(f"标注比例: {labeled_ratio:.1%}")
    log.info(f"对比学习: {'✅ 启用' if module_params['use_contrastive'] else '❌ 禁用'}")
    log.info(f"不确定性估计: {'✅ 启用' if module_params['use_uncertainty'] else '❌ 禁用'}")
    log.info(f"设备: {config['trainer']['accelerator']}")
    log.info("=" * 80)
    
    # 开始训练
    log.info("🎬 开始Swin Transformer + 对比学习 + 不确定性估计训练...")
    trainer.fit(module, datamodule=datamodule)
    
    # 测试最佳模型
    log.info("🧪 测试最佳模型...")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")
    
    log.info("🎉 训练完成！")
    log.info(f"📁 结果保存在: outputs/{exp_name}")
    
    # 打印最佳指标
    if hasattr(trainer.checkpoint_callback, 'best_model_score'):
        best_score = trainer.checkpoint_callback.best_model_score
        log.info(f"🏆 最佳验证IoU: {best_score:.4f}")


if __name__ == "__main__":
    main()

