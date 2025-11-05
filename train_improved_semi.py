#!/usr/bin/env python3
"""
改进的半监督学习训练脚本
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

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from baseg.datamodules_semi import SemiSupervisedEMSDataModule
from baseg.modules.semi_supervised import SemiSupervisedModule


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
    
    log.info(f"确定性训练环境设置完成 (seed={seed})")


def main():
    """主训练函数"""
    # 设置确定性训练
    set_deterministic_training(42)
    
    # 配置文件路径
    config_path = "configs/semi_supervised/ems_segformer-mit-b3_semi_improved.py"
    
    # 加载配置
    log.info(f"加载配置文件: {config_path}")
    config = Config.fromfile(config_path)
    
    # 设置实验名称（添加时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['name']}_labeled{int(config['data']['labeled_ratio']*100)}pct_{timestamp}"
    config["name"] = exp_name
    log.info(f"实验名称: {exp_name}")
    log.info(f"时间戳: {timestamp}")
    
    # 创建数据模块
    log.info("创建改进的半监督数据模块...")
    datamodule = SemiSupervisedEMSDataModule(**config["data"])
    
    # 创建模型模块
    log.info("创建改进的半监督学习模块...")
    model_config = config["model"]
    semi_config = config["semi_supervised"]
    loss = config.get("loss", "bce")
    
    module = SemiSupervisedModule(
        model_config,
        loss=loss,
        pseudo_threshold=semi_config["pseudo_threshold"],
        consistency_weight=semi_config["consistency_weight"],
        pseudo_weight=semi_config["pseudo_weight"],
        ramp_up_epochs=semi_config["ramp_up_epochs"],
        ema_decay=semi_config["ema_decay"],
        use_ema_teacher=semi_config["use_ema_teacher"],
    )
    
    # 初始化预训练权重
    module.init_pretrained()
    log.info("预训练权重初始化完成")
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="outputs",
        name=exp_name,
        default_hp_metric=False,
        log_graph=True,
    )
    
    # 记录超参数
    logger.log_hyperparams({
        "model_type": model_config["type"],
        "backbone": model_config["backbone"]["type"],
        "loss": loss,
        "labeled_ratio": config["data"]["labeled_ratio"],
        "labeled_batch_size": config["data"]["labeled_batch_size"],
        "unlabeled_batch_size": config["data"]["unlabeled_batch_size"],
        "patch_size": config["data"]["patch_size"],
        "pseudo_threshold": semi_config["pseudo_threshold"],
        "consistency_weight": semi_config["consistency_weight"],
        "pseudo_weight": semi_config["pseudo_weight"],
        "ramp_up_epochs": semi_config["ramp_up_epochs"],
        "ema_decay": semi_config["ema_decay"],
        "max_epochs": config["trainer"]["max_epochs"],
    })
    
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
    
    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"outputs/{exp_name}/version_0/weights",
        filename="best-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 创建训练器
    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=callbacks,
    )
    
    # 开始训练
    log.info("开始改进的半监督学习训练...")
    trainer.fit(module, datamodule=datamodule)
    
    # 测试最佳模型
    log.info("测试最佳模型...")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")
    
    log.info("训练完成！")


if __name__ == "__main__":
    main()






