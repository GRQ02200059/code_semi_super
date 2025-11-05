#!/usr/bin/env python3
"""
不确定性估计模块集成测试脚本
验证所有组件是否正确配置
"""

import os
import sys
import torch
from pathlib import Path
from loguru import logger as log

# 添加src到路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mmengine import Config
from baseg.modules.semi_supervised_uncertainty import SemiSupervisedUncertaintyModule
from baseg.modules.uncertainty import (
    UncertaintyEstimator,
    UncertaintyAwarePseudoLabeling
)


def test_uncertainty_estimator():
    """测试不确定性估计器"""
    log.info("=" * 60)
    log.info("测试1: 不确定性估计器")
    log.info("=" * 60)
    
    try:
        # 创建估计器
        estimator = UncertaintyEstimator(
            n_samples=5,
            min_threshold=0.8,
            max_threshold=0.99,
            uncertainty_weight=0.1
        )
        log.info("✅ 不确定性估计器创建成功")
        
        # 创建模拟模型
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)
                self.dropout = torch.nn.Dropout2d(0.1)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.dropout(x)
                return x
        
        model = DummyModel()
        model.eval()
        
        # 创建模拟输入
        x = torch.randn(2, 3, 64, 64)
        
        # 估计不确定性
        mean_pred, uncertainty = estimator.estimate_uncertainty(model, x)
        
        log.info(f"  - 输入形状: {x.shape}")
        log.info(f"  - 均值预测形状: {mean_pred.shape}")
        log.info(f"  - 不确定性形状: {uncertainty.shape}")
        log.info(f"  - 平均不确定性: {uncertainty.mean().item():.6f}")
        log.info(f"  - 最大不确定性: {uncertainty.max().item():.6f}")
        
        # 测试自适应阈值
        adaptive_t = estimator.adaptive_threshold(uncertainty, base_threshold=0.9)
        log.info(f"  - 自适应阈值形状: {adaptive_t.shape}")
        log.info(f"  - 自适应阈值范围: [{adaptive_t.min().item():.4f}, {adaptive_t.max().item():.4f}]")
        
        log.info("✅ 不确定性估计器测试通过")
        return True
        
    except Exception as e:
        log.error(f"❌ 不确定性估计器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pseudo_labeling():
    """测试不确定性感知伪标签生成"""
    log.info("=" * 60)
    log.info("测试2: 不确定性感知伪标签生成")
    log.info("=" * 60)
    
    try:
        # 创建估计器
        estimator = UncertaintyEstimator(n_samples=5)
        
        # 创建伪标签生成器
        pseudo_labeling = UncertaintyAwarePseudoLabeling(
            estimator=estimator,
            base_threshold=0.9,
            use_adaptive_threshold=True,
            log_uncertainty=True
        )
        log.info("✅ 不确定性感知伪标签生成器创建成功")
        
        # 创建模拟模型
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)
                self.dropout = torch.nn.Dropout2d(0.1)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.dropout(x)
                return x
        
        model = DummyModel()
        
        # 创建模拟输入
        x = torch.randn(2, 3, 64, 64)
        
        # 生成伪标签
        pseudo_labels, confidence_mask, uncertainty = pseudo_labeling.generate_pseudo_labels(model, x)
        
        log.info(f"  - 输入形状: {x.shape}")
        log.info(f"  - 伪标签形状: {pseudo_labels.shape}")
        log.info(f"  - 置信度掩码形状: {confidence_mask.shape}")
        log.info(f"  - 不确定性形状: {uncertainty.shape}")
        log.info(f"  - 高置信度像素比例: {confidence_mask.mean().item():.2%}")
        log.info(f"  - 伪标签正样本比例: {pseudo_labels.mean().item():.2%}")
        
        # 获取统计信息
        stats = pseudo_labeling.get_uncertainty_stats()
        log.info(f"  - 统计信息: {stats}")
        
        log.info("✅ 不确定性感知伪标签生成测试通过")
        return True
        
    except Exception as e:
        log.error(f"❌ 不确定性感知伪标签生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_integration():
    """测试完整模块集成"""
    log.info("=" * 60)
    log.info("测试3: 完整模块集成")
    log.info("=" * 60)
    
    try:
        # 加载配置
        config_path = "configs/semi_supervised/ems_swin_semi_uncertainty.py"
        log.info(f"加载配置: {config_path}")
        config = Config.fromfile(config_path)
        
        # 创建模块
        model_config = config["model"]
        semi_config = config["semi_supervised"]
        
        module = SemiSupervisedUncertaintyModule(
            model_config,
            loss="bce",
            pseudo_threshold=semi_config["pseudo_threshold"],
            consistency_weight=semi_config["consistency_weight"],
            pseudo_weight=semi_config["pseudo_weight"],
            ramp_up_epochs=semi_config["ramp_up_epochs"],
            ema_decay=semi_config["ema_decay"],
            use_ema_teacher=semi_config["use_ema_teacher"],
            use_contrastive=semi_config.get("use_contrastive", True),
            contrastive_weight=semi_config.get("contrastive_weight", 0.5),
            contrastive_temperature=semi_config.get("contrastive_temperature", 0.07),
            contrastive_mode=semi_config.get("contrastive_mode", "global"),
            projection_dim=semi_config.get("projection_dim", 128),
            projection_hidden_dim=semi_config.get("projection_hidden_dim", 256),
            use_uncertainty=semi_config.get("use_uncertainty", True),
            uncertainty_n_samples=semi_config.get("uncertainty_n_samples", 5),
            uncertainty_weight=semi_config.get("uncertainty_weight", 0.1),
            min_threshold=semi_config.get("min_threshold", 0.8),
            max_threshold=semi_config.get("max_threshold", 0.99),
            use_adaptive_threshold=semi_config.get("use_adaptive_threshold", True),
        )
        log.info("✅ 模块创建成功")
        
        # 检查关键组件
        log.info("检查模块组件:")
        log.info(f"  - 模型: {type(module.model).__name__}")
        log.info(f"  - 使用EMA教师: {module.use_ema_teacher}")
        if module.use_ema_teacher:
            log.info(f"  - 教师模型: {type(module.teacher_model).__name__}")
        log.info(f"  - 使用对比学习: {module.use_contrastive}")
        if module.use_contrastive:
            log.info(f"  - 投影头: {type(module.projection_head).__name__}")
            log.info(f"  - 对比学习模式: {module.contrastive_mode}")
        log.info(f"  - 使用不确定性估计: {module.use_uncertainty}")
        if module.use_uncertainty:
            log.info(f"  - 不确定性估计器: {type(module.uncertainty_estimator).__name__}")
            log.info(f"  - 伪标签生成器: {type(module.uncertainty_pseudo_labeling).__name__}")
            log.info(f"  - 自适应阈值: {module.use_adaptive_threshold}")
        
        # 测试前向传播
        log.info("测试前向传播:")
        x = torch.randn(2, 4, 128, 128)  # 4通道输入（模拟多模态）
        
        with torch.no_grad():
            output = module.model(x)
        
        log.info(f"  - 输入形状: {x.shape}")
        log.info(f"  - 输出形状: {output.shape}")
        
        # 测试伪标签生成（使用不确定性）
        if module.use_uncertainty:
            log.info("测试不确定性感知伪标签生成:")
            with torch.no_grad():
                pseudo_labels, confidence_mask, uncertainty = module._generate_pseudo_labels_with_uncertainty(
                    x,
                    use_teacher=False
                )
            
            log.info(f"  - 伪标签形状: {pseudo_labels.shape}")
            log.info(f"  - 置信度掩码形状: {confidence_mask.shape}")
            log.info(f"  - 不确定性形状: {uncertainty.shape}")
            log.info(f"  - 高置信度像素: {confidence_mask.sum().item()}/{confidence_mask.numel()} "
                    f"({confidence_mask.mean().item():.2%})")
            log.info(f"  - 平均不确定性: {uncertainty.mean().item():.6f}")
        
        log.info("✅ 模块集成测试通过")
        return True
        
    except Exception as e:
        log.error(f"❌ 模块集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    log.info("🧪 开始不确定性估计模块集成测试")
    log.info("")
    
    results = []
    
    # 测试1: 不确定性估计器
    results.append(("不确定性估计器", test_uncertainty_estimator()))
    log.info("")
    
    # 测试2: 伪标签生成
    results.append(("伪标签生成", test_pseudo_labeling()))
    log.info("")
    
    # 测试3: 模块集成
    results.append(("模块集成", test_module_integration()))
    log.info("")
    
    # 打印测试结果摘要
    log.info("=" * 60)
    log.info("测试结果摘要")
    log.info("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        log.info(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    log.info("=" * 60)
    if all_passed:
        log.info("🎉 所有测试通过！不确定性估计模块已正确集成")
        log.info("")
        log.info("下一步:")
        log.info("  1. 运行训练脚本: python train_swin_uncertainty.py")
        log.info("  2. 或使用启动脚本: bash start_uncertainty_training.sh")
        return 0
    else:
        log.error("❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())

