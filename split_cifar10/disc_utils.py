# 双判别器与 slice 工具（与 split_mnist/exp32 一致，feat_dim 可配）
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TASKS = 5


class FeatDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class SliceDiscriminator(nn.Module):
    def __init__(self, input_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TaskFromSliceHead(nn.Module):
    """从 slice-max 向量预测任务 ID，用于无 replay 下将 Task-IL 转化为 Class-IL（推理时路由）。"""
    def __init__(self, num_tasks=5, hidden=32):
        super().__init__()
        self.num_tasks = num_tasks
        self.net = nn.Sequential(
            nn.Linear(num_tasks, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_tasks),
        )

    def forward(self, slice_max_vec):
        return self.net(slice_max_vec)


def slice_max_vector(logits, num_tasks, device, pad_to=NUM_TASKS):
    B = logits.size(0)
    vec = [logits[:, 2*k:2*k+2].max(dim=1).values for k in range(num_tasks)]
    out = torch.stack(vec, dim=1)
    if num_tasks < pad_to:
        out = F.pad(out, (0, pad_to - num_tasks), value=0.0)
    return out
