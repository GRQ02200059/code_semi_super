from datetime import datetime
from functools import partial
from pathlib import Path
import sys
import os
import torch
import numpy as np
import random

# 动态添加当前项目的src目录到Python路径
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 全局种子常量
SEED = 42


def setup_reproducible_training(seed: int = SEED):
    """设置随机种子确保结果可重现"""
    # 设置环境变量确保完全的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置所有随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置日志
from loguru import logger as log

# 设置更详细的日志格式
log.remove()
log.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
# 添加文件日志
os.makedirs("logs", exist_ok=True)
log.add(
    "logs/training_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="500 MB",
)

from argdantic import ArgField, ArgParser
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.callbacks

from baseg.datamodules import EMSDataModule
from baseg.io import read_raster_profile, write_raster
from baseg.modules import MultiTaskModule, SingleTaskModule
from baseg.tiling import SmoothTiler
from baseg.utils import exp_name_timestamp, find_best_checkpoint

cli = ArgParser()


@cli.command()
def train(
    cfg_path: Path = ArgField("-c", description="Path to the config file."),
    keep_name: bool = ArgField(
        "-k", default=False, description="Keep the experiment name as specified in the config file."
    ),
):
    # 设置随机种子确保结果可重现
    setup_reproducible_training(SEED)
    seed_everything(SEED, workers=True)
    log.info(f"Random seed set to {SEED} for reproducibility")
    
    log.info(f"Loading config from: {cfg_path}")
    config = Config.fromfile(cfg_path)
    # set the experiment name
    assert "name" in config, "Experiment name not specified in config."
    exp_name = exp_name_timestamp(config["name"]) if not keep_name else config["name"]
    config["name"] = exp_name
    log.info(f"Experiment name: {exp_name}")

    # datamodule
    log.info("Preparing the data module...")
    # 如果配置中没有seed参数，则添加默认值
    if "seed" not in config["data"]:
        config["data"]["seed"] = SEED
    
    # 检查是否为半监督学习配置
    if "labeled_ratio" in config["data"] or "consistency_augmentation" in config["data"]:
        from baseg.datamodules_semi import SemiSupervisedEMSDataModule
        datamodule = SemiSupervisedEMSDataModule(**config["data"])
        log.info("Using SemiSupervisedEMSDataModule for semi-supervised learning")
    else:
        datamodule = EMSDataModule(**config["data"])
        log.info("Using EMSDataModule for supervised learning")
    
    log.info(f"Data configuration: {config['data']}")

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    log.info(f"Model configuration: {model_config['type']}")
    loss = config["loss"] if "loss" in config else "bce"
    log.info(f"Using loss function: {loss}")
    
    # 检查是否为半监督学习配置
    if "labeled_ratio" in config["data"] or "consistency_augmentation" in config["data"]:
        from baseg.modules.semi_supervised import SemiSupervisedModule
        module_class = SemiSupervisedModule
        log.info("Using SemiSupervisedModule for semi-supervised learning")
        
        # 半监督模块需要额外的参数
        semi_config = config.get("semi_supervised", {})
        module = module_class(
            model_config,
            loss=loss,
            pseudo_threshold=semi_config.get("pseudo_threshold", 0.95),
            consistency_weight=semi_config.get("consistency_weight", 1.0),
            pseudo_weight=semi_config.get("pseudo_weight", 1.0),
            ramp_up_epochs=semi_config.get("ramp_up_epochs", 10),
            ema_decay=semi_config.get("ema_decay", 0.99),
            use_ema_teacher=semi_config.get("use_ema_teacher", True),
        )
    else:
        # 普通监督学习
        module_class = MultiTaskModule if "aux_classes" in model_config["decode_head"] else SingleTaskModule
        log.info(f"Using {module_class.__name__} for supervised learning")
        module = module_class(model_config, loss=loss, mask_lc=config["mask_lc"]) if "mask_lc" in config else module_class(model_config, loss=loss)
    
    module.init_pretrained()
    log.info("Model initialized with pretrained weights")

    log.info("Preparing the trainer...")
    # 创建增强的TensorBoard记录器
    logger = TensorBoardLogger(
        save_dir="outputs",
        name=exp_name,
        default_hp_metric=False,
        log_graph=True,  # 记录模型结构
    )
    
    # 添加超参数记录
    logger.log_hyperparams({
        "model_type": model_config["type"],
        "backbone": model_config["backbone"]["type"],
        "loss": loss,
        "batch_size": config["data"]["batch_size_train"],
        "patch_size": config["data"]["patch_size"],
        "learning_rate": 1e-4,  # 从模块中获取
        "weight_decay": 1e-4,
        "max_epochs": config["trainer"]["max_epochs"],
    })
    
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
    log.info(f"Config saved to {config_dir / 'config.py'}")
    
    # 增加更详细的回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights" / "loss",
            monitor="val_loss",
            mode="min",
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,  # 保存val_loss最小的3个模型
            every_n_epochs=1,
            verbose=True,
        ),
        # 添加进度条回调
        pytorch_lightning.callbacks.RichProgressBar(
            refresh_rate=1,
            leave=True,
        ),
        # 添加学习率监控
        pytorch_lightning.callbacks.LearningRateMonitor(
            logging_interval='step',
            log_momentum=True,
        ),
        # 添加早停回调
        pytorch_lightning.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=True,
            mode='min',
        ),
        # 添加模型摘要回调
        pytorch_lightning.callbacks.ModelSummary(
            max_depth=2
        ),
        # 保存val_iou最高的3个模型
        pytorch_lightning.callbacks.ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights" / "iou",
            monitor="val_iou",
            mode="max",
            filename="best-{epoch:02d}-{val_iou:.4f}",
            save_top_k=3,
            verbose=True,
        ),
    ]
    
    # 设置详细的日志记录
    trainer_config = config["trainer"].copy()
    trainer_config.update({
        "log_every_n_steps": 10,  # 每10步记录一次日志
        "enable_progress_bar": True,
        "enable_model_summary": True,
    })
    
    trainer = Trainer(**trainer_config, callbacks=callbacks, logger=logger)
    log.info(f"Trainer configuration: {trainer_config}")
    log.info(f"Total epochs: {trainer_config.get('max_epochs', 'Not specified')}")
    log.info(f"Training on device: {trainer_config.get('accelerator', 'cpu')}")

    log.info("Starting the training...")
    trainer.fit(module, datamodule=datamodule)


@cli.command()
def test(
    exp_path: Path = ArgField("-e", description="Path to the experiment folder."),
    checkpoint: Path = ArgField(
        "-c",
        default=None,
        description="Path to the checkpoint file. If not specified, the best checkpoint will be loaded.",
    ),
    predict: bool = ArgField(default=False, description="Generate predictions on the test set."),
):
    # 设置随机种子确保结果可重现
    setup_reproducible_training(SEED)
    seed_everything(SEED, workers=True)
    log.info(f"Random seed set to {SEED} for reproducibility")
    
    log.info(f"Loading experiment from: {exp_path}")
    # 自动向上查找config.py
    config_dir = exp_path
    while not (config_dir / "config.py").exists() and config_dir != config_dir.parent:
        config_dir = config_dir.parent
    config_path = config_dir / "config.py"
    models_path = exp_path
    # asserts to check the experiment folders
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"
    # load training config
    config = Config.fromfile(config_path)

    # datamodule
    log.info("Preparing the data module...")
    # 如果配置中没有seed参数，则添加默认值
    if "seed" not in config["data"]:
        config["data"]["seed"] = SEED
    
    # 检查是否为半监督学习配置
    if "labeled_ratio" in config["data"] or "consistency_augmentation" in config["data"]:
        from baseg.datamodules_semi import SemiSupervisedEMSDataModule
        datamodule = SemiSupervisedEMSDataModule(**config["data"])
        log.info("Using SemiSupervisedEMSDataModule for semi-supervised learning")
    else:
        datamodule = EMSDataModule(**config["data"])
        log.info("Using EMSDataModule for supervised learning")

    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    log.info(f"Using checkpoint: {checkpoint}")

    module_opts = dict(config=config["model"])
    loss = config["loss"] if "loss" in config else "bce"
    module_opts.update( loss=loss)
    if predict:
        tiler = SmoothTiler(
            tile_size=config["data"]["patch_size"],
            batch_size=config["data"]["batch_size_eval"],
            channels_first=True,
            mirrored=False,
        )
        output_path = exp_path / "predictions"
        output_path.mkdir(parents=True, exist_ok=True)
        inference_fn = partial(process_inference, output_path=output_path)
        module_opts.update(tiler=tiler, predict_callback=inference_fn)

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    
    # 检查是否为半监督学习配置
    if "labeled_ratio" in config["data"] or "consistency_augmentation" in config["data"]:
        from baseg.modules.semi_supervised import SemiSupervisedModule
        module_class = SemiSupervisedModule
        log.info("Using SemiSupervisedModule for semi-supervised learning")
        
        # 半监督模块需要额外的参数
        semi_config = config.get("semi_supervised", {})
        module_opts = dict(
            config=config["model"],
            loss=config.get("loss", "bce"),
            pseudo_threshold=semi_config.get("pseudo_threshold", 0.95),
            consistency_weight=semi_config.get("consistency_weight", 1.0),
            pseudo_weight=semi_config.get("pseudo_weight", 1.0),
            ramp_up_epochs=semi_config.get("ramp_up_epochs", 10),
            ema_decay=semi_config.get("ema_decay", 0.99),
            use_ema_teacher=semi_config.get("use_ema_teacher", True),
        )
    else:
        # 普通监督学习
        module_class = MultiTaskModule if "aux_classes" in model_config["decode_head"] else SingleTaskModule
        log.info(f"Using {module_class.__name__} for supervised learning")
    
    module = module_class.load_from_checkpoint(checkpoint, **module_opts)

    logger = TensorBoardLogger(save_dir="outputs", name=config["name"], version=exp_path.stem)
    if predict:
        log.info("Generating predictions...")
        trainer = Trainer(**config["evaluation"], logger=False)
        trainer.predict(module, datamodule=datamodule, return_predictions=False)
    else:
        log.info("Starting the testing...")
        trainer = Trainer(**config["evaluation"], logger=logger)
        trainer.test(module, datamodule=datamodule)


@cli.command()
def test_multi(
    root: Path = ArgField("-r", description="Path to the root folder of the experiments."),
    from_date: datetime = ArgField("-f", default=None, description="Start date for the experiments to test."),
    epoch: int = ArgField("-e", default=None, description="Number of epochs to test."),
):
    assert root.exists(), f"Root folder does not exist: {root}"
    experiments = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]

    for exp_path in experiments:
        exp_path = exp_path / "version_0"
        log.info(f"Testing experiment: {exp_path}")
        config_path = exp_path / "config.py"
        weights_path = exp_path / "weights"

        if not config_path.exists():
            log.warning(f"Config file not found in: {config_path}")
            continue
        if not weights_path.exists():
            log.warning(f"Models folder not found in: {weights_path}")
            continue

        # parse timestamp with format <name_with_underscores>_<date>_<time>, exclude the name
        timestamp = "_".join(exp_path.parent.stem.split("_")[-2:])
        timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        if from_date and timestamp < from_date:
            log.info(f"Skipping experiment from {timestamp}, too old.")
            continue

        checkpoint = None
        if epoch is not None:
            checkpoint = list(weights_path.glob(f"model-epoch={epoch}*.ckpt"))
            if not checkpoint:
                log.warning(f"Checkpoint not found for epoch {epoch} in: {weights_path}")
                continue
            checkpoint = checkpoint[0]

        test.callback(exp_path, checkpoint=checkpoint)


def process_inference(
    batch: dict,
    output_path: Path,
):
    assert output_path.exists(), f"Output path does not exist: {output_path}"
    # for binary segmentation
    prediction = (batch["pred"] > 0.5).int().unsqueeze(0)
    prediction = prediction.cpu().numpy()
    # store the prediction as a GeoTIFF, reading the spatial information from the input image
    image_path = Path(batch["metadata"]["S2L2A"][0])
    input_profile = read_raster_profile(image_path)
    output_profile = input_profile.copy()
    output_profile.update(dtype="uint8", count=1)
    output_file = output_path / f"{image_path.stem}.tif"
    write_raster(path=output_file, data=prediction, profile=output_profile)


if __name__ == "__main__":
    # 在程序入口处设置随机种子
    setup_reproducible_training(SEED)
    cli()
