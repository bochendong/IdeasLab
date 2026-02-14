# Split CIFAR-10 数据加载（5 任务，每任务 2 类）
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .constants import TASK_DIR


def get_cifar10_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2470, 0.2435, 0.2616],
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2470, 0.2435, 0.2616],
        ),
    ])


def make_task_dataloader(
    task_num: int,
    data_dir: str,
    train: bool = True,
    batch_size: int = 128,
    task_dir: dict = None,
):
    """创建单个 task 的 DataLoader"""
    task_dir = task_dir or TASK_DIR
    tf = get_cifar10_transform(train=train)
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tf)
    labels = set(task_dir[task_num])
    idx = [i for i, (_, y) in enumerate(ds) if int(y) in labels]
    sub = Subset(ds, idx)
    return DataLoader(
        sub, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=2, pin_memory=True,
    )


def build_all_task_loaders(data_dir: str, batch_size: int = 128, task_dir: dict = None):
    """构建所有 5 个 task 的 train/test DataLoader"""
    task_dir = task_dir or TASK_DIR
    loaders = {}
    for t in range(len(task_dir)):
        loaders[t] = (
            make_task_dataloader(t, data_dir, train=True, batch_size=batch_size, task_dir=task_dir),
            make_task_dataloader(t, data_dir, train=False, batch_size=batch_size, task_dir=task_dir),
        )
    return loaders
