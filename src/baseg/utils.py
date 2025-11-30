from datetime import datetime
from pathlib import Path


def exp_name_timestamp(name: str) -> str:
    """Generates a name for the experiment starting with the given name and
    appending the current date and time for uniqueness.

    Args:
        name (str): The name of the experiment.

    Returns:
        str: The name of the experiment with the current date and time appended.
    """
    now = datetime.now()
    return f"{name}_{now.strftime('%Y%m%d_%H%M%S')}"


def find_best_checkpoint(ckpt_path: Path, metric: str, mode: str = "min") -> Path:
    """Finds the best checkpoint in the given path based on the given metric.

    Args:
        ckpt_path (Path): The path to the checkpoint directory.
        metric (str): The metric to use for comparison.
        mode (str, optional): The mode to use for comparison. Defaults to "min".

    Returns:
        Path: The path to the best checkpoint.
    """
    assert ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}"
    assert mode in ["min", "max"], f"Invalid mode: {mode}"
    
    # 定义可能的checkpoint存储路径
    search_paths = [
        ckpt_path,  # 直接在根目录
        ckpt_path / "weights" / "loss",  # loss最佳模型
        ckpt_path / "weights" / "iou",   # iou最佳模型
    ]
    
    # find the best checkpoint
    best_ckpt = None
    best_value = None
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        for ckpt in search_path.glob("*.ckpt"):
            if ckpt.stem == "last":
                continue
            try:
                value = float(ckpt.stem.split("=")[-1])
                if best_value is None or (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
                    best_ckpt = ckpt
                    best_value = value
            except (ValueError, IndexError):
                # 跳过无法解析的checkpoint文件名
                continue
    
    assert best_ckpt is not None, f"No checkpoint found in: {ckpt_path} or its subdirectories (weights/loss, weights/iou)"
    return best_ckpt
