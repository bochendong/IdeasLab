# Split CIFAR-100 评估：TAw / TAg 矩阵，与 AdaGauss 一致
import numpy as np
import torch

from .constants import NUM_TASKS, CLASSES_PER_TASK


@torch.no_grad()
def eval_task_taw_tag(model, dataloader, task_u, current_num_tasks, device):
    """
    在任务 u 的 dataloader 上计算 TAw 和 TAg 准确率。
    model: 输出 logits (B, 100)，其中 logits[:, u*10:(u+1)*10] 为任务 u 的 head。
    task_u: 0..current_num_tasks-1
    current_num_tasks: 已学任务数，只使用前 current_num_tasks 个 head（共 current_num_tasks*10 维）。
    返回: (acc_taw, acc_tag) 标量 0..1
    """
    model.eval()
    correct_taw, correct_tag, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, 100) 或 (B, num_heads*10)
        # 只取已学类的 logits
        max_cls = current_num_tasks * CLASSES_PER_TASK
        if logits.size(1) > max_cls:
            logits = logits[:, :max_cls]

        # TAw: 已知是任务 u，只用 head u
        offset = task_u * CLASSES_PER_TASK
        logits_u = logits[:, offset : offset + CLASSES_PER_TASK]
        pred_u = logits_u.argmax(dim=1) + offset
        correct_taw += (pred_u == y).sum().item()

        # TAg: 在所有已学类上 argmax
        pred_all = logits.argmax(dim=1)
        correct_tag += (pred_all == y).sum().item()
        total += y.size(0)
    total = max(total, 1)
    return correct_taw / total, correct_tag / total


def compute_acc_matrices(model, task_loaders, current_task_t, device):
    """
    学完任务 current_task_t 后，对 u in [0, current_task_t] 在任务 u 的测试集上算 TAw/TAg。
    task_loaders[t] = (train_loader, test_loader)。
    返回: (acc_taw_row, acc_tag_row)，长度为 current_task_t+1。
    """
    acc_taw_row = []
    acc_tag_row = []
    for u in range(current_task_t + 1):
        _, test_loader = task_loaders[u]
        at, ag = eval_task_taw_tag(
            model, test_loader, u, current_task_t + 1, device
        )
        acc_taw_row.append(at)
        acc_tag_row.append(ag)
    return acc_taw_row, acc_tag_row


def compute_forgetting(acc_matrix):
    """每个任务 k 的遗忘量：历史最好 - 最终。acc_matrix[t][k] = 学完 t 后对任务 k 的准确率。"""
    T = len(acc_matrix) - 1
    forget = {}
    for k in range(T + 1):
        best = max(acc_matrix[t][k] for t in range(k, T + 1))
        final = acc_matrix[T][k]
        forget[k] = best - final
    return forget


def compute_bwt(acc_matrix):
    """Backward Transfer: (1/(T)) * sum_k (acc[T][k] - acc[k][k])"""
    T = len(acc_matrix) - 1
    if T <= 0:
        return 0.0
    vals = [acc_matrix[T][k] - acc_matrix[k][k] for k in range(T)]
    return sum(vals) / len(vals)
