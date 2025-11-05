#!/usr/bin/env python3
"""
半监督学习训练脚本
使用伪标签和一致性正则化进行燃烧区域分割
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import random
from loguru import logger as log

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from mmengine import Config

from baseg.datamodules_semi import SemiSupervisedEMSDataModule
from baseg.modules.semi_supervised import SemiSupervisedModule


def set_deterministic_training():
    """设置确定性训练环境"""
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 设置所有随机种子
    seed_everything(42, workers=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    log.info("确定性训练环境设置完成 (seed=42)")


def main():
    """主训练函数"""
    # 设置确定性训练
    set_deterministic_training()
    
    # 配置文件路径
    config_path = "configs/semi_supervised/ems_segformer-mit-b3_semi_50ep.py"
    
    # 加载配置
    log.info(f"加载配置文件: {config_path}")
    config = Config.fromfile(config_path)
    
    # 设置实验名称
    exp_name = f"{config['name']}_labeled{int(config['data']['labeled_ratio']*100)}pct"
    config["name"] = exp_name
    log.info(f"实验名称: {exp_name}")
    
    # 创建数据模块
    log.info("创建半监督数据模块...")
    datamodule = SemiSupervisedEMSDataModule(**config["data"])
    
    # 创建模型模块
    log.info("创建半监督学习模块...")
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
        "pseudo_threshold": semi_config["pseudo_threshold"],
        "consistency_weight": semi_config["consistency_weight"],
        "pseudo_weight": semi_config["pseudo_weight"],
        "ramp_up_epochs": semi_config["ramp_up_epochs"],
        "ema_decay": semi_config["ema_decay"],
        "use_ema_teacher": semi_config["use_ema_teacher"],
        "max_epochs": config["trainer"]["max_epochs"],
    })
    
    # 保存配置文件
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
    log.info(f"配置文件已保存到: {config_dir / 'config.py'}")
    
    # 设置回调函数
    callbacks = [
        # 模型检查点 - 基于验证损失
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights" / "val_loss",
            monitor="val_loss",
            mode="min",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            every_n_epochs=1,
            verbose=True,
        ),
        # 模型检查点 - 基于验证IoU
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights" / "val_iou",
            monitor="val_iou",
            mode="max",
            filename="best-{epoch:02d}-{val_iou:.4f}",
            save_top_k=3,
            every_n_epochs=1,
            verbose=True,
        ),
        # 早停
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
            strict=False,
        ),
    ]
    
    # 创建训练器
    trainer = Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        precision=config["trainer"]["precision"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=config["trainer"]["check_val_every_n_epoch"],
    )
    
    # 开始训练
    log.info("开始半监督训练...")
    log.info(f"训练参数: labeled_ratio={config['data']['labeled_ratio']}, "
             f"pseudo_threshold={semi_config['pseudo_threshold']}, "
             f"consistency_weight={semi_config['consistency_weight']}")
    
    trainer.fit(module, datamodule)
    
    # 训练完成
    log.info("半监督训练完成!")
    log.info(f"最佳模型保存在: {logger.log_dir}")
    
    # 自动测试最佳模型
    log.info("开始测试最佳模型...")
    best_model_path = callbacks[0].best_model_path
    if best_model_path:
        trainer.test(module, datamodule, ckpt_path=best_model_path)
        log.info(f"测试完成，使用模型: {best_model_path}")
    else:
        trainer.test(module, datamodule)
        log.info("测试完成，使用当前模型")


if __name__ == "__main__":
    main()








