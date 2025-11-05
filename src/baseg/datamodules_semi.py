from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import pickle
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from baseg.datasets import EMSCropDataset, EMSImageDataset
from baseg.samplers import SequentialTiledSampler
from baseg.samplers.single import RandomTiledSampler


class SemiSupervisedEMSDataset(Dataset):
    """半监督学习数据集包装器"""
    
    def __init__(
        self,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        labeled_batch_size: int,
        unlabeled_batch_size: int,
        consistency_augmentation: bool = True,
        seed: int = 42,
    ):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.consistency_augmentation = consistency_augmentation
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 计算数据集长度（以较大的数据集为准）
        self.length = max(len(labeled_dataset), len(unlabeled_dataset))
        
        # 一致性增强变换
        if consistency_augmentation:
            self.consistency_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.1, contrast_limit=0.1),
                ToTensorV2(),
            ], additional_targets={
                "S2L2A": "image",
                "DEL": "mask",
                "CM": "mask", 
                "ESA_LC": "mask",
            })
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 生成一个混合批次，包含标注和无标注样本
        batch = {
            "labeled_samples": [],
            "unlabeled_samples": []
        }
        
        # 采样标注数据
        for i in range(self.labeled_batch_size):
            labeled_idx = (idx * self.labeled_batch_size + i) % len(self.labeled_dataset)
            try:
                labeled_sample = self.labeled_dataset[labeled_idx]
            except (AttributeError, TypeError):
                # 如果是Subset，直接通过索引访问
                labeled_sample = self.labeled_dataset.dataset[self.labeled_dataset.indices[labeled_idx]]
            batch["labeled_samples"].append(labeled_sample)
        
        # 采样无标注数据
        for i in range(self.unlabeled_batch_size):
            unlabeled_idx = (idx * self.unlabeled_batch_size + i) % len(self.unlabeled_dataset)
            try:
                unlabeled_sample = self.unlabeled_dataset[unlabeled_idx]
            except (AttributeError, TypeError):
                # 如果是Subset，直接通过索引访问
                unlabeled_sample = self.unlabeled_dataset.dataset[self.unlabeled_dataset.indices[unlabeled_idx]]
            
            # 为无标注数据移除标签（如果存在）
            unlabeled_clean = {
                "S2L2A": unlabeled_sample["S2L2A"],
                "metadata": unlabeled_sample.get("metadata", {})
            }
            
            # 如果启用一致性增强，为无标注数据创建增强版本
            if self.consistency_augmentation:
                # 转换为numpy格式进行增强
                if isinstance(unlabeled_sample["S2L2A"], torch.Tensor):
                    image_np = unlabeled_sample["S2L2A"].permute(1, 2, 0).numpy()
                else:
                    image_np = unlabeled_sample["S2L2A"]
                
                # 确保数据格式正确
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # 应用一致性增强
                try:
                    augmented = self.consistency_transform(image=image_np)
                    unlabeled_clean["S2L2A_aug"] = augmented["image"]
                except Exception as e:
                    print(f"一致性增强失败: {e}")
                    # 如果增强失败，使用原图
                    unlabeled_clean["S2L2A_aug"] = unlabeled_sample["S2L2A"]
            
            batch["unlabeled_samples"].append(unlabeled_clean)
        
        return batch


class SemiSupervisedEMSDataModule(LightningDataModule):
    """半监督学习数据模块"""
    
    transform_targets = {
        "S2L2A": "image",
        "DEL": "mask",
        "CM": "mask",
        "GRA": "mask",
        "ESA_LC": "mask",
    }

    def __init__(
        self,
        root: Optional[Path] = None,  # 兼容原配置文件
        labeled_root: Optional[Path] = None,
        unlabeled_root: Optional[Path] = None,
        patch_size: int = 512,
        modalities: List[str] = ["S2L2A", "DEL", "ESA_LC", "CM"],
        labeled_batch_size: int = 4,
        unlabeled_batch_size: int = 4,
        batch_size_eval: int = 16,
        batch_size_train: int = 8,  # 兼容参数
        num_workers: int = 4,
        labeled_ratio: float = 0.1,  # 有标注数据的比例
        consistency_augmentation: bool = True,
        seed: int = 42,
        **kwargs  # 接受其他参数
    ) -> None:
        super().__init__()
        
        # 处理参数兼容性
        if root is not None and labeled_root is None:
            labeled_root = root  # 如果提供了root参数，使用它作为labeled_root
        
        self.labeled_root = labeled_root or "data/ems"  # 默认路径
        self.unlabeled_root = unlabeled_root or self.labeled_root  # 如果没有指定，使用相同的根目录
        self.modalities = modalities
        self.patch_size = patch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.labeled_ratio = labeled_ratio
        self.consistency_augmentation = consistency_augmentation
        self.seed = seed
        
        # 数据分割保存路径（用于可复现性）
        self.split_cache_dir = Path(self.labeled_root) / "splits_cache"
        self.split_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子 (修复albumentations的种子设置)
        import random
        random.seed(seed)
        np.random.seed(seed)
        # Albumentations使用numpy的随机状态，所以设置numpy种子即可
        
        # 训练时的强增强（有标注数据）
        self.labeled_train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(rotate_limit=360, value=0, mask_value=255, p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.02, contrast_limit=0.02),
            A.GaussNoise(var_limit=(0.0, 0.005), p=0.2),
            ToTensorV2(),
        ], additional_targets=self.transform_targets)
        
        # 训练时的弱增强（无标注数据）
        self.unlabeled_train_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.01, contrast_limit=0.01),
            ToTensorV2(),
        ], additional_targets=self.transform_targets)
        
        # 评估时的变换
        self.eval_transform = A.Compose([
            ToTensorV2()
        ], additional_targets=self.transform_targets)

    def _generate_split_indices(self, total_size: int, labeled_size: int) -> Tuple[List[int], List[int]]:
        """生成数据分割索引（使用固定随机种子确保可复现）"""
        # 设置随机种子
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # 生成随机排列
        indices = torch.randperm(total_size).tolist()
        labeled_indices = sorted(indices[:labeled_size])  # 排序以便验证
        unlabeled_indices = sorted(indices[labeled_size:])
        
        return labeled_indices, unlabeled_indices
    
    def _split_labeled_unlabeled(self, dataset_class, subset: str, transform):
        """将数据集分割为有标注和无标注部分（支持固定分割以实现可复现性）"""
        # 创建完整数据集（先获取数据集大小，不加载数据）
        # 为了生成唯一的分割文件名，我们需要先创建数据集实例来获取大小
        full_dataset = dataset_class(
            root=self.labeled_root,
            subset=subset,
            modalities=self.modalities,
            transform=transform,
        )
        
        # 计算有标注数据的数量
        total_size = len(full_dataset)
        labeled_size = int(total_size * self.labeled_ratio)
        unlabeled_size = total_size - labeled_size
        
        # 生成唯一的分割文件名（基于seed, ratio, subset等）
        split_filename = f"split_{subset}_ratio{self.labeled_ratio:.2f}_seed{self.seed}_size{total_size}.pkl"
        split_filepath = self.split_cache_dir / split_filename
        
        # 尝试加载已保存的分割结果
        if split_filepath.exists():
            print(f"[数据分割] 从缓存加载固定分割: {split_filepath}")
            with open(split_filepath, 'rb') as f:
                split_data = pickle.load(f)
                labeled_indices = split_data['labeled_indices']
                unlabeled_indices = split_data['unlabeled_indices']
                
            # 验证分割结果是否匹配
            if len(labeled_indices) != labeled_size or len(unlabeled_indices) != unlabeled_size:
                print(f"[数据分割] 警告: 缓存的分割大小不匹配，重新生成分割")
                labeled_indices, unlabeled_indices = self._generate_split_indices(total_size, labeled_size)
                # 保存新的分割结果
                split_data = {
                    'labeled_indices': labeled_indices,
                    'unlabeled_indices': unlabeled_indices,
                    'seed': self.seed,
                    'labeled_ratio': self.labeled_ratio,
                    'total_size': total_size,
                }
                with open(split_filepath, 'wb') as f:
                    pickle.dump(split_data, f)
                print(f"[数据分割] 已保存新的固定分割: {split_filepath}")
        else:
            # 生成新的分割
            print(f"[数据分割] 创建新的固定分割（seed={self.seed}, ratio={self.labeled_ratio:.2f}）")
            labeled_indices, unlabeled_indices = self._generate_split_indices(total_size, labeled_size)
            
            # 保存分割结果
            split_data = {
                'labeled_indices': labeled_indices,
                'unlabeled_indices': unlabeled_indices,
                'seed': self.seed,
                'labeled_ratio': self.labeled_ratio,
                'total_size': total_size,
            }
            with open(split_filepath, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"[数据分割] 已保存固定分割到: {split_filepath}")
        
        # 创建子数据集的包装器，兼容采样器
        class SubsetDatasetWrapper:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                # 处理IndexedBounds对象
                if hasattr(idx, 'index'):
                    # 来自采样器的IndexedBounds对象
                    return self.dataset[idx]
                else:
                    # 普通整数索引，转换为实际数据集索引
                    actual_idx = self.indices[idx]
                    return self.dataset[actual_idx]
            
            def image_shapes(self):
                full_shapes = self.dataset.image_shapes()
                return [full_shapes[i] for i in self.indices]
        
        labeled_dataset = SubsetDatasetWrapper(full_dataset, labeled_indices)
        
        # 为无标注数据集创建特殊的数据集（移除标签）
        unlabeled_full_dataset = dataset_class(
            root=self.unlabeled_root,
            subset=subset,
            modalities=["S2L2A"],  # 只加载图像，不加载标签
            transform=self.unlabeled_train_transform,
        )
        unlabeled_dataset = SubsetDatasetWrapper(unlabeled_full_dataset, unlabeled_indices)
        
        return labeled_dataset, unlabeled_dataset

    def setup(self, stage=None):
        if stage == "fit":
            # 分割训练数据
            self.labeled_train_set, self.unlabeled_train_set = self._split_labeled_unlabeled(
                EMSCropDataset, "train", self.labeled_train_transform
            )
            
            # 创建半监督训练数据集
            self.semi_train_set = SemiSupervisedEMSDataset(
                labeled_dataset=self.labeled_train_set,
                unlabeled_dataset=self.unlabeled_train_set,
                labeled_batch_size=self.labeled_batch_size,
                unlabeled_batch_size=self.unlabeled_batch_size,
                consistency_augmentation=self.consistency_augmentation,
                seed=self.seed,
            )
            
            print(f"半监督训练数据设置完成:")
            print(f"  - 有标注数据: {len(self.labeled_train_set)} 样本")
            print(f"  - 无标注数据: {len(self.unlabeled_train_set)} 样本")
            print(f"  - 标注比例: {self.labeled_ratio:.1%}")
            print(f"  - 半监督数据集长度: {len(self.semi_train_set)}")
            
            # 验证集（全部有标注）
            self.val_set = EMSCropDataset(
                root=self.labeled_root,
                subset="val",
                modalities=self.modalities,
                transform=self.eval_transform,
            )
            
            print(f"训练数据: 有标注={len(self.labeled_train_set)}, "
                  f"无标注={len(self.unlabeled_train_set)}, "
                  f"验证={len(self.val_set)}")
            
        elif stage == "test":
            self.test_set = EMSCropDataset(
                root=self.labeled_root,
                subset="test",
                modalities=self.modalities,
                transform=self.eval_transform,
            )
            
        elif stage == "predict":
            self.pred_set = EMSImageDataset(
                root=self.labeled_root,
                subset="test",
                modalities=self.modalities,
                transform=self.eval_transform,
            )

    def train_dataloader(self):
        """半监督训练数据加载器"""
        # 暂时简化为标准监督学习训练，使用采样器
        from baseg.samplers.single import RandomTiledSampler
        
        # 使用标注数据集进行训练
        return DataLoader(
            self.labeled_train_set,
            sampler=RandomTiledSampler(self.labeled_train_set, tile_size=self.patch_size, seed=self.seed),
            batch_size=self.labeled_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def _collate_fn(self, batch):
        """自定义批次整理函数"""
        # batch是一个列表，每个元素包含labeled_samples和unlabeled_samples
        labeled_samples = []
        unlabeled_samples = []
        
        for sample in batch:
            labeled_samples.extend(sample['labeled_samples'])
            unlabeled_samples.extend(sample['unlabeled_samples'])
        
        # 使用默认的collate函数处理每个部分
        from torch.utils.data.dataloader import default_collate
        
        try:
            result = {
                'labeled': default_collate(labeled_samples) if labeled_samples else None,
                'unlabeled': default_collate(unlabeled_samples) if unlabeled_samples else None
            }
            return result
        except Exception as e:
            print(f"Collate function error: {e}")
            # 返回空结果作为fallback
            return {
                'labeled': None,
                'unlabeled': None
            }

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            sampler=SequentialTiledSampler(
                self.val_set,
                tile_size=self.patch_size,
            ),
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            sampler=SequentialTiledSampler(
                self.test_set,
                tile_size=self.patch_size,
            ),
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_set,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class UnlabeledEMSDataset(EMSCropDataset):
    """无标注数据集（只加载图像，不加载标签）"""
    
    def __init__(self, *args, **kwargs):
        # 强制只加载图像模态
        kwargs['modalities'] = ['S2L2A']
        super().__init__(*args, **kwargs)
    
    def _preprocess(self, sample: dict) -> dict:
        """只处理图像，不处理标签"""
        sample["image"] = np.clip(sample.pop("S2L2A").transpose(1, 2, 0), 0, 1)
        return sample
        
    def _postprocess(self, sample: dict) -> dict:
        """只返回图像"""
        sample["S2L2A"] = sample.pop("image")
        return sample
