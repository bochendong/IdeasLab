# Split CIFAR-100 常量（10 任务 × 10 类，与 AdaGauss / iCaRL 一致）

NUM_TASKS = 10
CLASSES_PER_TASK = 10

# 与 AdaGauss dataset_config 中 cifar100_icarl 的 class_order 一致，便于对比
CIFAR100_ICARL_CLASS_ORDER = [
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
    28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
    98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
    36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33,
]

# 任务 t 对应的原始 CIFAR-100 类标：TASK_CLASSES[t] = [c0, c1, ..., c9]
TASK_CLASSES = {
    t: CIFAR100_ICARL_CLASS_ORDER[t * CLASSES_PER_TASK : (t + 1) * CLASSES_PER_TASK]
    for t in range(NUM_TASKS)
}

# 全局类标 -> 任务内类标 (0..9)：label_global in [0..99] -> (task_id, label_in_task)
def global_to_task_label(global_label: int):
    """全局类标 (0..99，按 class_order 重排后) -> (task_id, 任务内 0..9)"""
    for t in range(NUM_TASKS):
        if global_label < (t + 1) * CLASSES_PER_TASK:
            return t, global_label - t * CLASSES_PER_TASK
    return NUM_TASKS - 1, CLASSES_PER_TASK - 1
