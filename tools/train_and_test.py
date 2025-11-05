#!/usr/bin/env python
import os
import subprocess
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def get_latest_experiment(base_dir="outputs"):
    """获取最新的实验目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    experiments = [d for d in base_path.iterdir() if d.is_dir()]
    if not experiments:
        return None
    
    # 按修改时间排序，获取最新的
    latest_exp = max(experiments, key=lambda x: x.stat().st_mtime)
    # 检查是否有version_0子目录
    version_dir = latest_exp / "version_0"
    if version_dir.exists():
        return version_dir
    return latest_exp

def run_test(exp_path, predict=False):
    """运行测试"""
    if not exp_path:
        log.error("找不到实验目录，无法进行测试")
        return False
    
    log.info(f"找到实验目录: {exp_path}")
    
    # 构建测试命令
    test_cmd = ["python", "tools/launch.py", "test", "-e", str(exp_path)]
    if predict:
        test_cmd.append("--predict")
    
    # 执行测试
    log.info(f"开始测试: {' '.join(test_cmd)}")
    result = subprocess.run(test_cmd)
    
    if result.returncode != 0:
        log.error("测试失败")
        return False
    
    log.info("测试完成")
    return True

def main():
    parser = argparse.ArgumentParser(description="训练模型并在训练结束后自动测试")
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--predict", action="store_true", help="是否生成预测结果")
    parser.add_argument("--nohup", action="store_true", help="是否使用nohup后台运行")
    parser.add_argument("--test-only", action="store_true", help="仅测试最新的实验")
    args = parser.parse_args()
    
    if args.test_only:
        exp_path = get_latest_experiment()
        run_test(exp_path, args.predict)
        return
    
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"配置文件不存在: {config_path}")
        return
    
    # 构建训练命令
    train_cmd = ["python", "tools/launch.py", "train", "-c", str(config_path)]
    
    # 如果需要后台运行，使用nohup
    if args.nohup:
        log_file = f"logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 构建完整的命令，包括训练和测试
        full_cmd = " ".join(train_cmd) + f" > {log_file} 2>&1"
        
        # 添加测试命令
        test_cmd = f" && python {os.path.abspath(__file__)} --test-only"
        if args.predict:
            test_cmd += " --predict"
        
        full_cmd = f"nohup bash -c '{full_cmd}{test_cmd}' &"
        
        log.info(f"后台运行训练和测试命令，日志将保存到: {log_file}")
        os.system(full_cmd)
        log.info("训练和测试已在后台启动...")
        return
    
    # 前台运行训练
    log.info(f"开始训练: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    
    if result.returncode != 0:
        log.error("训练失败，退出")
        return
    
    log.info("训练完成，准备测试...")
    
    # 获取最新的实验目录
    exp_path = get_latest_experiment()
    run_test(exp_path, args.predict)

if __name__ == "__main__":
    main() 