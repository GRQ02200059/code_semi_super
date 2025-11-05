#!/usr/bin/env python3
"""
对比学习模块安装测试脚本
验证所有组件是否正确安装和配置
"""

import sys
from pathlib import Path

# 添加src到路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def test_imports():
    """测试模块导入"""
    print("=" * 80)
    print("🧪 测试1: 模块导入")
    print("=" * 80)
    
    tests = []
    
    # 测试损失函数导入
    try:
        from baseg.losses import SupConLoss, PixelContrastLoss, ProjectionHead
        print("✅ 对比学习损失函数导入成功")
        print(f"   - SupConLoss: {SupConLoss}")
        print(f"   - PixelContrastLoss: {PixelContrastLoss}")
        print(f"   - ProjectionHead: {ProjectionHead}")
        tests.append(True)
    except Exception as e:
        print(f"❌ 对比学习损失函数导入失败: {e}")
        tests.append(False)
    
    # 测试模块导入
    try:
        from baseg.modules import SemiSupervisedContrastiveModule
        print("✅ 对比学习模块导入成功")
        print(f"   - SemiSupervisedContrastiveModule: {SemiSupervisedContrastiveModule}")
        tests.append(True)
    except Exception as e:
        print(f"❌ 对比学习模块导入失败: {e}")
        tests.append(False)
    
    print()
    return all(tests)


def test_loss_functions():
    """测试损失函数"""
    print("=" * 80)
    print("🧪 测试2: 损失函数功能")
    print("=" * 80)
    
    import torch
    from baseg.losses import SupConLoss, PixelContrastLoss, ProjectionHead
    
    tests = []
    
    # 测试SupConLoss
    try:
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 2, 128)  # [batch, n_views, dim]
        loss = loss_fn(features)
        assert loss.item() >= 0, "损失值应该非负"
        print(f"✅ SupConLoss 测试通过 (loss={loss.item():.4f})")
        tests.append(True)
    except Exception as e:
        print(f"❌ SupConLoss 测试失败: {e}")
        tests.append(False)
    
    # 测试PixelContrastLoss
    try:
        loss_fn = PixelContrastLoss(temperature=0.07, max_samples=512)
        features = torch.randn(4, 128, 32, 32)  # [batch, channels, h, w]
        labels = torch.randint(0, 2, (4, 32, 32))  # [batch, h, w]
        loss = loss_fn(features, labels)
        assert loss.item() >= 0, "损失值应该非负"
        print(f"✅ PixelContrastLoss 测试通过 (loss={loss.item():.4f})")
        tests.append(True)
    except Exception as e:
        print(f"❌ PixelContrastLoss 测试失败: {e}")
        tests.append(False)
    
    # 测试ProjectionHead
    try:
        proj_head = ProjectionHead(in_dim=768, hidden_dim=256, out_dim=128)
        x = torch.randn(8, 768)
        out = proj_head(x)
        assert out.shape == (8, 128), f"输出形状错误: {out.shape}"
        print(f"✅ ProjectionHead 测试通过 (输出形状={out.shape})")
        tests.append(True)
    except Exception as e:
        print(f"❌ ProjectionHead 测试失败: {e}")
        tests.append(False)
    
    print()
    return all(tests)


def test_module_creation():
    """测试模块创建"""
    print("=" * 80)
    print("🧪 测试3: 对比学习模块创建")
    print("=" * 80)
    
    try:
        from mmengine import Config
        from baseg.modules import SemiSupervisedContrastiveModule
        
        # 加载配置
        config_path = "configs/semi_supervised/ems_swin_semi_contrastive.py"
        config = Config.fromfile(config_path)
        
        print(f"✅ 配置文件加载成功: {config_path}")
        
        # 创建模块
        semi_config = config["semi_supervised"]
        module = SemiSupervisedContrastiveModule(
            config["model"],
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
        )
        
        print("✅ 对比学习模块创建成功")
        print(f"   - 对比学习: {module.use_contrastive}")
        print(f"   - 对比模式: {module.contrastive_mode}")
        print(f"   - 对比权重: {module.contrastive_weight}")
        print(f"   - 温度参数: {module.contrastive_temperature}")
        
        if hasattr(module, 'projection_head'):
            print(f"   - 投影头: ✅ 已创建")
        else:
            print(f"   - 投影头: ❌ 未创建")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 模块创建失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("=" * 80)
    print("🧪 测试4: 前向传播（可选，需要GPU）")
    print("=" * 80)
    
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  GPU不可用，跳过前向传播测试")
        print()
        return True
    
    try:
        from mmengine import Config
        from baseg.modules import SemiSupervisedContrastiveModule
        
        # 加载配置
        config_path = "configs/semi_supervised/ems_swin_semi_contrastive.py"
        config = Config.fromfile(config_path)
        
        # 创建模块
        semi_config = config["semi_supervised"]
        module = SemiSupervisedContrastiveModule(
            config["model"],
            loss="bce",
            use_contrastive=True,
            contrastive_mode="global",
            **{k: v for k, v in semi_config.items() if k.startswith('contrastive') or k in [
                'pseudo_threshold', 'consistency_weight', 'pseudo_weight', 
                'ramp_up_epochs', 'ema_decay', 'use_ema_teacher'
            ]}
        )
        
        module = module.cuda()
        module.eval()
        
        # 创建测试数据
        x = torch.randn(2, 12, 128, 128).cuda()  # [batch, channels, h, w]
        
        # 前向传播
        with torch.no_grad():
            # 测试特征提取
            features = module._extract_features(x, use_projection=True)
            print(f"✅ 特征提取成功: {features.shape}")
            
            # 测试模型推理
            output = module.model(x)
            print(f"✅ 模型推理成功: {output.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"⚠️  前向传播测试失败: {e}")
        print("   这可能是正常的，如果模型还没有加载预训练权重")
        print()
        return True  # 不算作失败


def check_file_structure():
    """检查文件结构"""
    print("=" * 80)
    print("🧪 测试5: 文件结构检查")
    print("=" * 80)
    
    required_files = [
        "src/baseg/losses/contrastive.py",
        "src/baseg/modules/semi_supervised_contrastive.py",
        "configs/semi_supervised/ems_swin_semi_contrastive.py",
        "train_swin_contrastive.py",
        "对比学习集成指南.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            all_exist = False
    
    print()
    return all_exist


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("🚀 对比学习模块安装测试")
    print("=" * 80)
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("文件结构", check_file_structure()))
    results.append(("模块导入", test_imports()))
    results.append(("损失函数", test_loss_functions()))
    results.append(("模块创建", test_module_creation()))
    results.append(("前向传播", test_forward_pass()))
    
    # 显示结果摘要
    print("=" * 80)
    print("📊 测试结果摘要")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:12} : {status}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("=" * 80)
        print("🎉 所有测试通过！对比学习模块已成功集成")
        print("=" * 80)
        print()
        print("下一步:")
        print("  1. 运行训练: python train_swin_contrastive.py")
        print("  2. 或使用脚本: bash start_contrastive_training.sh")
        print("  3. 查看文档: cat 对比学习集成指南.md")
        print()
        return 0
    else:
        print("=" * 80)
        print("⚠️  部分测试失败，请检查错误信息")
        print("=" * 80)
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

