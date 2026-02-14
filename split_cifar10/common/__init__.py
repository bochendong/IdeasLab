# Split CIFAR-10 公共模块
from .constants import TASK_DIR, NUM_TASKS
from .data import get_cifar10_transform, make_task_dataloader, build_all_task_loaders
from .eval import (
    cal_acc_class_il,
    cal_acc_class_il_routed,
    cal_acc_task_il,
    evaluate_after_task,
    compute_forgetting,
    compute_bwt,
)

__all__ = [
    "TASK_DIR",
    "NUM_TASKS",
    "get_cifar10_transform",
    "make_task_dataloader",
    "build_all_task_loaders",
    "cal_acc_class_il",
    "cal_acc_class_il_routed",
    "cal_acc_task_il",
    "evaluate_after_task",
    "compute_forgetting",
    "compute_bwt",
]
