#!/usr/bin/env python3
"""
测试随机种子设置是否正确的脚本
"""

import os
import sys
import numpy as np
import torch
import random
import albumentations as A
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent / "src"))

from baseg.samplers.single import RandomTiledSampler
from baseg.datamodules import EMSDataModule


def test_numpy_seed():
    """测试numpy随机种子"""
    print("=== 测试 NumPy 随机种子 ===")
    
    # 第一次运行
    np.random.seed(42)
    result1 = np.random.random(5)
    print(f"第一次运行: {result1}")
    
    # 第二次运行
    np.random.seed(42)
    result2 = np.random.random(5)
    print(f"第二次运行: {result2}")
    
    if np.allclose(result1, result2):
        print("✅ NumPy 随机种子设置正确")
    else:
        print("❌ NumPy 随机种子设置有问题")
    print()


def test_torch_seed():
    """测试PyTorch随机种子"""
    print("=== 测试 PyTorch 随机种子 ===")
    
    # 第一次运行
    torch.manual_seed(42)
    result1 = torch.randn(5)
    print(f"第一次运行: {result1}")
    
    # 第二次运行
    torch.manual_seed(42)
    result2 = torch.randn(5)
    print(f"第二次运行: {result2}")
    
    if torch.allclose(result1, result2):
        print("✅ PyTorch 随机种子设置正确")
    else:
        print("❌ PyTorch 随机种子设置有问题")
    print()


def test_albumentations_seed():
    """测试Albumentations随机种子"""
    print("=== 测试 Albumentations 随机种子 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 定义变换
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),  # 设置p=1.0确保变换总是应用
        A.RandomRotate90(p=1.0),
    ])
    
    # 第一次运行
    A.random.seed(42)
    result1 = transform(image=test_image)['image']
    
    # 第二次运行
    A.random.seed(42)
    result2 = transform(image=test_image)['image']
    
    if np.array_equal(result1, result2):
        print("✅ Albumentations 随机种子设置正确")
    else:
        print("❌ Albumentations 随机种子设置有问题")
    print()


def test_random_sampler():
    """测试RandomTiledSampler的种子设置"""
    print("=== 测试 RandomTiledSampler 随机种子 ===")
    
    # 创建一个模拟数据集类
    class MockDataset:
        def __len__(self):
            return 10
            
        def image_shapes(self):
            return [(256, 256) for _ in range(10)]
    
    dataset = MockDataset()
    
    # 第一次运行
    sampler1 = RandomTiledSampler(dataset, tile_size=64, length=20, seed=42)
    indices1 = sampler1.indices.copy()
    
    # 第二次运行
    sampler2 = RandomTiledSampler(dataset, tile_size=64, length=20, seed=42)
    indices2 = sampler2.indices.copy()
    
    if np.array_equal(indices1, indices2):
        print("✅ RandomTiledSampler 随机种子设置正确")
    else:
        print("❌ RandomTiledSampler 随机种子设置有问题")
        print(f"第一次: {indices1[:5]}...")
        print(f"第二次: {indices2[:5]}...")
    print()


def test_environment_variables():
    """测试环境变量设置"""
    print("=== 测试环境变量 ===")
    
    expected_vars = {
        'PYTHONHASHSEED': '42',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
    }
    
    for var, expected_value in expected_vars.items():
        actual_value = os.environ.get(var)
        if actual_value == expected_value:
            print(f"✅ {var} = {actual_value}")
        else:
            print(f"❌ {var} = {actual_value} (期望: {expected_value})")
    print()


if __name__ == "__main__":
    print("开始测试随机种子设置的可重现性...\n")
    
    test_environment_variables()
    test_numpy_seed()
    test_torch_seed()
    test_albumentations_seed()
    test_random_sampler()
    
    print("测试完成！")

