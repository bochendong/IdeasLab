# Split CIFAR-100 (10x10) - 与 AdaGauss 一致的评估结构：TAw / TAg

from .constants import (
    NUM_TASKS,
    CLASSES_PER_TASK,
    CIFAR100_ICARL_CLASS_ORDER,
    TASK_CLASSES,
)
from . import data
from . import eval

__all__ = [
    "NUM_TASKS",
    "CLASSES_PER_TASK",
    "CIFAR100_ICARL_CLASS_ORDER",
    "TASK_CLASSES",
    "data",
    "eval",
]
