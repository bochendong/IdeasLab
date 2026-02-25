# Split CIFAR-100 数据加载（10×10，与 AdaGauss 相同 class_order 与预处理）
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from .constants import NUM_TASKS, CLASSES_PER_TASK, CIFAR100_ICARL_CLASS_ORDER

# CIFAR-100 常用归一化（与 AdaGauss dataset_config 一致）
CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2009, 0.1984, 0.2023)


def get_cifar100_transform(train: bool, pad: int = 4, crop: int = 32):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(crop, padding=pad),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _remap_labels(targets, class_order):
    """将原始 CIFAR-100 类标 (0..99 任意顺序) 映射为 class_order 下的 0..99。"""
    order = np.array(class_order)
    targets = np.asarray(targets)
    remapped = np.zeros_like(targets, dtype=np.int64)
    for new_idx, old_idx in enumerate(order):
        remapped[targets == old_idx] = new_idx
    return remapped.tolist()


def build_all_task_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    class_order: list = None,
):
    """
    构建 10 个任务的 (train_loader, test_loader)。
    data_dir: 存放 CIFAR-100 的目录（如 data/cifar100_100）。
    class_order: 100 个类的顺序，默认 CIFAR100_ICARL_CLASS_ORDER。
    返回: loaders[t] = (train_loader, test_loader)，样本的 y 为 0..99（任务 t 对应 10*t..10*t+9）。
    """
    class_order = class_order or CIFAR100_ICARL_CLASS_ORDER
    trn_tf = get_cifar100_transform(train=True)
    tst_tf = get_cifar100_transform(train=False)

    trn_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=trn_tf)
    tst_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tst_tf)

    # 统一用 numpy 取 target 并重映射
    trn_y = np.array(trn_ds.targets)
    tst_y = np.array(tst_ds.targets)
    trn_remap = _remap_labels(trn_y, class_order)
    tst_remap = _remap_labels(tst_y, class_order)
    trn_ds.targets = trn_remap
    tst_ds.targets = tst_remap

    loaders = {}
    for t in range(NUM_TASKS):
        low, high = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
        trn_idx = [i for i, y in enumerate(trn_remap) if low <= y < high]
        tst_idx = [i for i, y in enumerate(tst_remap) if low <= y < high]
        trn_sub = Subset(trn_ds, trn_idx)
        tst_sub = Subset(tst_ds, tst_idx)
        loaders[t] = (
            DataLoader(trn_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(tst_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )
    return loaders
