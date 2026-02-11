# Split MNIST 评估：Task-IL / Class-IL / Forgetting / BWT
import numpy as np
import torch

from .constants import TASK_DIR


@torch.no_grad()
def cal_acc_class_il(model, dataloader, valid_out_dim, device=None):
    """Class-IL 准确率：在 valid_out_dim 个类上做分类"""
    device = device or next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        logits = logits[:, :valid_out_dim]
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def cal_acc_task_il(model, dataloader, task_id, task_dir=None, device=None):
    """Task-IL 准确率：已知 task 时，在 task 对应类上做分类"""
    task_dir = task_dir or TASK_DIR
    device = device or next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    cls = task_dir[task_id]
    cls_tensor = torch.tensor(cls, device=device)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        logits2 = logits[:, cls]
        pred2 = logits2.argmax(dim=1)
        pred = cls_tensor[pred2]
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(total, 1)


def evaluate_after_task(model, task_done, task_loaders, task_dir=None, device=None):
    """
    完成 task_done 后，评估所有已见 task 的 Task-IL 和 Class-IL。
    返回 (task_il, class_il, avg_task, avg_class)
    """
    task_dir = task_dir or TASK_DIR
    device = device or next(model.parameters()).device
    valid_out_dim = 2 * (task_done + 1)
    task_il, class_il = {}, {}
    for t in range(task_done + 1):
        test_loader = task_loaders[t][1]
        task_il[t] = cal_acc_task_il(model, test_loader, t, task_dir=task_dir, device=device)
        class_il[t] = cal_acc_class_il(model, test_loader, valid_out_dim, device=device)
    avg_task = float(np.mean(list(task_il.values())))
    avg_class = float(np.mean(list(class_il.values())))
    return task_il, class_il, avg_task, avg_class


def compute_forgetting(acc_matrix):
    """每个 task 的遗忘量：历史最好 - 最终"""
    T = len(acc_matrix) - 1
    forget = {}
    for k in range(T + 1):
        best = max(acc_matrix[t][k] for t in range(k, T + 1))
        final = acc_matrix[T][k]
        forget[k] = best - final
    return forget


def compute_bwt(acc_matrix):
    """Backward Transfer：训练新 task 后对旧 task 的影响"""
    T = len(acc_matrix) - 1
    vals = []
    for k in range(T):
        vals.append(acc_matrix[T][k] - acc_matrix[k][k])
    return float(sum(vals) / len(vals)) if vals else 0.0
