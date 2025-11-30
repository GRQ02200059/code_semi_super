#!/usr/bin/env python3
"""
测试所有checkpoint脚本
自动测试实验目录中的所有checkpoint并生成对比报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger as log

# 添加src到路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import random
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything

from baseg.datamodules_semi import SemiSupervisedEMSDataModule
from baseg.modules.semi_supervised import SemiSupervisedModule
from baseg.modules.semi_supervised_contrastive import SemiSupervisedContrastiveModule
from baseg.modules.semi_supervised_uncertainty import SemiSupervisedUncertaintyModule


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_everything(seed, workers=True)


def find_checkpoints(exp_dir):
    """查找实验目录中的所有checkpoint"""
    exp_path = Path(exp_dir)
    
    # 查找所有.ckpt文件
    checkpoints = []
    
    # IoU最佳模型
    iou_dir = exp_path / "version_0" / "weights" / "iou"
    if iou_dir.exists():
        for ckpt in sorted(iou_dir.glob("*.ckpt")):
            checkpoints.append({
                'path': str(ckpt),
                'name': ckpt.name,
                'type': 'iou',
                'category': 'IoU最佳模型'
            })
    
    # Loss最佳模型
    loss_dir = exp_path / "version_0" / "weights" / "loss"
    if loss_dir.exists():
        for ckpt in sorted(loss_dir.glob("*.ckpt")):
            checkpoints.append({
                'path': str(ckpt),
                'name': ckpt.name,
                'type': 'loss',
                'category': 'Loss最佳模型'
            })
    
    return checkpoints


def test_checkpoint(ckpt_info, exp_dir, datamodule):
    """测试单个checkpoint"""
    log.info(f"=" * 80)
    log.info(f"测试: {ckpt_info['name']}")
    log.info(f"类型: {ckpt_info['category']}")
    log.info(f"=" * 80)
    
    try:
        # 加载配置
        config_path = Path(exp_dir) / "version_0" / "config.py"
        if config_path.exists():
            config = Config.fromfile(config_path)
        else:
            log.warning("配置文件不存在，使用默认配置")
            config = Config.fromfile("configs/semi_supervised/ems_swin_semi_contrastive.py")
        
        # 创建模型
        model_config = config["model"]
        semi_config = config.get("semi_supervised", {})
        loss = config.get("loss", "bce")
        
        # 根据配置自动选择模块类型
        use_contrastive = semi_config.get("use_contrastive", False)
        use_uncertainty = semi_config.get("use_uncertainty", False)
        
        if use_contrastive:
            log.info("使用 SemiSupervisedContrastiveModule")
            module = SemiSupervisedContrastiveModule(
                model_config,
                loss=loss,
                pseudo_threshold=semi_config.get("pseudo_threshold", 0.9),
                consistency_weight=semi_config.get("consistency_weight", 2.0),
                pseudo_weight=semi_config.get("pseudo_weight", 1.5),
                ramp_up_epochs=semi_config.get("ramp_up_epochs", 10),
                ema_decay=semi_config.get("ema_decay", 0.999),
                use_ema_teacher=semi_config.get("use_ema_teacher", True),
                use_contrastive=True,
                contrastive_weight=semi_config.get("contrastive_weight", 0.5),
                contrastive_temperature=semi_config.get("contrastive_temperature", 0.07),
                contrastive_mode=semi_config.get("contrastive_mode", "global"),
                projection_dim=semi_config.get("projection_dim", 128),
                projection_hidden_dim=semi_config.get("projection_hidden_dim", 256),
            )
        elif use_uncertainty:
            log.info("使用 SemiSupervisedUncertaintyModule")
            module = SemiSupervisedUncertaintyModule(
                model_config,
                loss=loss,
                pseudo_threshold=semi_config.get("pseudo_threshold", 0.9),
                consistency_weight=semi_config.get("consistency_weight", 2.0),
                pseudo_weight=semi_config.get("pseudo_weight", 1.5),
                ramp_up_epochs=semi_config.get("ramp_up_epochs", 10),
                ema_decay=semi_config.get("ema_decay", 0.999),
                use_ema_teacher=semi_config.get("use_ema_teacher", True),
                use_uncertainty=True,
                uncertainty_weight=semi_config.get("uncertainty_weight", 0.5),
            )
        else:
            log.info("使用基础 SemiSupervisedModule")
            module = SemiSupervisedModule(
                model_config,
                loss=loss,
                pseudo_threshold=semi_config.get("pseudo_threshold", 0.9),
                consistency_weight=semi_config.get("consistency_weight", 2.0),
                pseudo_weight=semi_config.get("pseudo_weight", 1.5),
                ramp_up_epochs=semi_config.get("ramp_up_epochs", 10),
                ema_decay=semi_config.get("ema_decay", 0.999),
                use_ema_teacher=semi_config.get("use_ema_teacher", True),
            )
        
        # 创建训练器
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            precision=16,
            logger=False,
        )
        
        # 测试
        results = trainer.test(
            module, 
            datamodule=datamodule, 
            ckpt_path=ckpt_info['path']
        )
        
        if results and len(results) > 0:
            result = results[0]
            log.info(f"✅ 测试完成!")
            log.info(f"   Test IoU: {result.get('test_iou', 0):.4f}")
            log.info(f"   Test F1:  {result.get('test_f1', 0):.4f}")
            log.info(f"   Test Loss: {result.get('test_loss', 0):.4f}")
            
            return {
                'checkpoint': ckpt_info['name'],
                'type': ckpt_info['type'],
                'category': ckpt_info['category'],
                'test_iou': result.get('test_iou', 0),
                'test_f1': result.get('test_f1', 0),
                'test_loss': result.get('test_loss', 0),
                'success': True
            }
        else:
            log.error("测试失败，无结果返回")
            return {
                'checkpoint': ckpt_info['name'],
                'type': ckpt_info['type'],
                'category': ckpt_info['category'],
                'success': False
            }
            
    except Exception as e:
        log.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'checkpoint': ckpt_info['name'],
            'type': ckpt_info['type'],
            'category': ckpt_info['category'],
            'error': str(e),
            'success': False
        }


def generate_report(results, exp_dir):
    """生成测试报告"""
    report_path = Path(exp_dir) / "test_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Checkpoint 测试报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"实验目录: {exp_dir}\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试数量: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        # 成功的测试
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        if successful:
            f.write("✅ 成功测试的Checkpoint:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Checkpoint':<50} {'Type':<10} {'IoU':>8} {'F1':>8} {'Loss':>8}\n")
            f.write("-" * 80 + "\n")
            
            for r in sorted(successful, key=lambda x: x.get('test_iou', 0), reverse=True):
                f.write(f"{r['checkpoint']:<50} {r['type']:<10} "
                       f"{r.get('test_iou', 0):>8.4f} "
                       f"{r.get('test_f1', 0):>8.4f} "
                       f"{r.get('test_loss', 0):>8.4f}\n")
            
            f.write("\n")
            
            # 最佳模型
            best_iou = max(successful, key=lambda x: x.get('test_iou', 0))
            best_f1 = max(successful, key=lambda x: x.get('test_f1', 0))
            best_loss = min(successful, key=lambda x: x.get('test_loss', float('inf')))
            
            f.write("🏆 最佳性能:\n")
            f.write("-" * 80 + "\n")
            f.write(f"最高 IoU:  {best_iou['checkpoint']} (IoU={best_iou.get('test_iou', 0):.4f})\n")
            f.write(f"最高 F1:   {best_f1['checkpoint']} (F1={best_f1.get('test_f1', 0):.4f})\n")
            f.write(f"最低 Loss: {best_loss['checkpoint']} (Loss={best_loss.get('test_loss', 0):.4f})\n")
            f.write("\n")
        
        if failed:
            f.write("❌ 失败的Checkpoint:\n")
            f.write("-" * 80 + "\n")
            for r in failed:
                f.write(f"{r['checkpoint']}: {r.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    log.info(f"📄 测试报告已保存: {report_path}")
    
    # 也保存JSON格式
    json_path = Path(exp_dir) / "test_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"📄 JSON报告已保存: {json_path}")
    
    return report_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试所有checkpoint')
    parser.add_argument('-e', '--experiment', type=str, required=True,
                       help='实验目录路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    exp_dir = args.experiment
    log.info(f"📂 实验目录: {exp_dir}")
    
    # 查找所有checkpoint
    log.info("🔍 查找checkpoint...")
    checkpoints = find_checkpoints(exp_dir)
    
    if not checkpoints:
        log.error("❌ 未找到任何checkpoint")
        return
    
    log.info(f"✅ 找到 {len(checkpoints)} 个checkpoint")
    for i, ckpt in enumerate(checkpoints, 1):
        log.info(f"   {i}. [{ckpt['category']}] {ckpt['name']}")
    
    # 加载配置创建数据模块
    log.info("\n📊 创建数据模块...")
    config_path = Path(exp_dir) / "version_0" / "config.py"
    if config_path.exists():
        config = Config.fromfile(config_path)
    else:
        log.warning("配置文件不存在，使用默认配置")
        config = Config.fromfile("configs/semi_supervised/ems_swin_semi_contrastive.py")
    
    datamodule = SemiSupervisedEMSDataModule(**config["data"])
    log.info("✅ 数据模块创建完成")
    
    # 测试所有checkpoint
    log.info(f"\n🧪 开始测试 {len(checkpoints)} 个checkpoint...")
    results = []
    
    for i, ckpt in enumerate(checkpoints, 1):
        log.info(f"\n[{i}/{len(checkpoints)}] 测试中...")
        result = test_checkpoint(ckpt, exp_dir, datamodule)
        results.append(result)
    
    # 生成报告
    log.info("\n📝 生成测试报告...")
    report_path = generate_report(results, exp_dir)
    
    # 显示摘要
    log.info("\n" + "=" * 80)
    log.info("📊 测试摘要")
    log.info("=" * 80)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        best_iou = max(successful, key=lambda x: x.get('test_iou', 0))
        log.info(f"✅ 成功: {len(successful)}/{len(results)}")
        log.info(f"🏆 最佳IoU: {best_iou.get('test_iou', 0):.4f} ({best_iou['checkpoint']})")
        log.info(f"📄 完整报告: {report_path}")
    else:
        log.error("❌ 所有测试都失败了")
    
    log.info("=" * 80)
    log.info("🎉 测试完成!")


if __name__ == "__main__":
    main()

