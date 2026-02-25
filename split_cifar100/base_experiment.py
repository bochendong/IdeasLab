# ===== Split CIFAR-100 基础实验（10×10，评估结构同 AdaGauss：TAw / TAg）=====
"""
与 MNIST 思路一致：多任务 head、正则等；评估与 AdaGauss 一致：acc_taw/acc_tag 矩阵、avg_accs、forgetting、BWT。
支持：slice margin、logit/特征蒸馏、EWC、SI。
"""
import os
import sys
import copy
import random
import logging
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_manager import ExperimentManager, get_output_dir
from split_cifar100.common import (
    NUM_TASKS,
    CLASSES_PER_TASK,
    build_all_task_loaders,
)
from split_cifar100.common.eval import (
    eval_task_taw_tag,
    compute_acc_matrices,
    compute_forgetting,
    compute_bwt,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "cifar100_100")
OUTPUT_DIR = get_output_dir("split_cifar100")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 与 AdaGauss 一致：backbone 输出特征维度 S=64
ADAGAUSS_FEAT_DIM = 64

DEFAULT_CONFIG = {
    "batch_size": 128,
    "epochs_per_task": 50,
    "lr": 0.1,
    "seed": 42,
    "num_workers": 4,
    "backbone": "resnet18",
    # 决策层 / 蒸馏（与 MNIST 结合）
    "lambda_slice_margin": 0.0,
    "slice_margin": 0.5,
    "lambda_logit_kd": 0.0,
    "tau_kd": 2.0,
    "lambda_feat_kd": 0.0,
    # 参数正则
    "ewc_lambda": 0.0,
    "si_lambda": 0.0,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(file_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(file_name), logging.StreamHandler()],
        force=True,
    )


# ------------------------
# 与 AdaGauss 一致的 ResNet18 + 多任务 head
# ------------------------
def build_model(backbone_name="resnet18", num_tasks=10, num_classes_per_task=10):
    """
    构建 10×10 分类模型。backbone="resnet18" 时与 AdaGauss 使用相同 backbone（ResNet18，特征 64 维）。
    """
    from split_cifar100.networks import resnet18_cifar

    feat_dim = ADAGAUSS_FEAT_DIM
    if backbone_name == "resnet18":
        backbone = resnet18_cifar(num_features=feat_dim)
        return ResNet18MultiHead(backbone, num_tasks, num_classes_per_task, feat_dim)
    # 轻量 CNN（不用于与 AdaGauss 对比时可用）
    return CIFAR100CNN(num_tasks=num_tasks, num_classes_per_task=num_classes_per_task)


class ResNet18MultiHead(nn.Module):
    """ResNet18 backbone（64 维特征）+ 10 个 head，与 AdaGauss 的 backbone 一致。"""

    def __init__(self, backbone, num_tasks, num_classes_per_task, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.feat_dim = feat_dim
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        logits = torch.cat([h(feat) for h in self.heads], dim=1)
        if return_feat:
            return logits, feat
        return logits


# ------------------------
# 简单 CNN（可选，不用于与 AdaGauss 对比）
# ------------------------
class CIFAR100CNN(nn.Module):
    """轻量 CNN + 10 个 head，用于快速试跑。"""

    def __init__(self, num_tasks=10, num_classes_per_task=10):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.feat_dim = 256 * 4 * 4
        self.heads = nn.ModuleList([
            nn.Linear(self.feat_dim, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, return_feat=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        feat = x.view(x.size(0), -1)
        logits = torch.cat([h(feat) for h in self.heads], dim=1)
        if return_feat:
            return logits, feat
        return logits


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


# ------------------------
# Slice margin / 蒸馏 / EWC / SI（与 MNIST 一致）
# ------------------------
def slice_margin_loss(logits, y, num_tasks, margin=0.5, device=DEVICE):
    """正确任务 slice 的 max 要 > 其它 slice 的 max + margin；每 slice 10 类。"""
    if num_tasks <= 1:
        return torch.tensor(0.0, device=device)
    B = logits.size(0)
    task_id = (y // CLASSES_PER_TASK).clamp(0, num_tasks - 1).long()
    correct_max = torch.zeros(B, device=device)
    for t in range(num_tasks):
        mask = task_id == t
        if mask.any():
            lo, hi = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
            correct_max[mask] = logits[mask, lo:hi].max(dim=1).values
    other_max_list = []
    for t in range(num_tasks):
        lo, hi = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
        other_max_list.append(logits[:, lo:hi].max(dim=1).values)
    other_max = torch.stack(other_max_list, dim=1)
    other_max[torch.arange(B, device=device), task_id] = -1e9
    other_max = other_max.max(dim=1).values
    gap = correct_max - other_max
    return F.relu(margin - gap).mean()


def logit_kd_loss(new_logits, old_logits, num_old_classes, tau=2.0, device=DEVICE):
    """对旧类 logits 做 soft CE 蒸馏（LwF 风格）。"""
    if num_old_classes <= 0:
        return torch.tensor(0.0, device=device)
    new_logits = new_logits[:, :num_old_classes] / tau
    old_logits = old_logits[:, :num_old_classes].detach() / tau
    old_probs = F.softmax(old_logits, dim=1)
    return -(old_probs * F.log_softmax(new_logits, dim=1)).sum(dim=1).mean()


def compute_fisher(model, task_loaders, task_id, valid_out_dim, device=DEVICE, num_samples=1000):
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p.data, device=device)
    train_loader = task_loaders[task_id][0]
    counted = 0
    for x, y in train_loader:
        if counted >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        if logits.dim() == 3:
            logits = logits[0]
        loss = F.cross_entropy(logits[:, :valid_out_dim], y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        counted += x.size(0)
    for n in fisher:
        fisher[n] /= max(counted, 1)
    return fisher


def ewc_penalty(model, ewc_fisher_star, device=DEVICE):
    loss = torch.tensor(0.0, device=device)
    for fisher, star in ewc_fisher_star:
        for n, p in model.named_parameters():
            if n in fisher and p.requires_grad:
                loss = loss + (fisher[n] * (p - star[n]) ** 2).sum()
    return loss


def si_penalty(model, si_omega_star_list, device=DEVICE):
    loss = torch.tensor(0.0, device=device)
    for omega, star in si_omega_star_list:
        for n, p in model.named_parameters():
            if n in omega and n in star and p.requires_grad:
                loss = loss + (omega[n] * (p - star[n]) ** 2).sum()
    return loss


def update_si_omega(si_omega, si_theta_prev, model, device=DEVICE):
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None or n not in si_omega:
            continue
        delta = (p.detach() - si_theta_prev[n]).to(device)
        si_omega[n] = si_omega[n].to(device) + (p.grad.detach() * delta).abs()
        si_theta_prev[n] = p.detach().clone()


# ------------------------
# 保存 AdaGauss 格式的 results（便于用 plot_adagauss_results 画图）
# ------------------------
def save_adagauss_format_results(exp_dir, acc_taw, acc_tag, timestamp=None):
    """acc_taw/acc_tag: (T, T) numpy，下三角有效。写入 results/ 下与 AdaGauss 同名的 txt。"""
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    T = acc_taw.shape[0]
    for name, mat in [("acc_taw", acc_taw), ("acc_tag", acc_tag)]:
        path = os.path.join(results_dir, f"{name}-{timestamp}.txt")
        with open(path, "w") as f:
            for t in range(T):
                row = "\t".join([f"{mat[t, u]:.6f}" if u <= t else "0.000000" for u in range(T)])
                f.write(row + "\n")
    # avg_accs: 每行一个数
    avg_taw = np.array([acc_taw[t, : t + 1].mean() for t in range(T)])
    avg_tag = np.array([acc_tag[t, : t + 1].mean() for t in range(T)])
    for name, vec in [("avg_accs_taw", avg_taw), ("avg_accs_tag", avg_tag)]:
        path = os.path.join(results_dir, f"{name}-{timestamp}.txt")
        with open(path, "w") as f:
            f.write("\t".join([f"{v:.6f}" for v in vec]) + "\n")
    for name, vec in [("wavg_accs_taw", avg_taw), ("wavg_accs_tag", avg_tag)]:
        path = os.path.join(results_dir, f"{name}-{timestamp}.txt")
        with open(path, "w") as f:
            f.write("\t".join([f"{v:.6f}" for v in vec]) + "\n")
    return timestamp


# ------------------------
# 主训练循环
# ------------------------
def run_experiment(
    run_name: str,
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])

    task_loaders = build_all_task_loaders(
        DATA_DIR,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )

    backbone_name = cfg.get("backbone", "resnet18")
    lambda_slice_margin = cfg.get("lambda_slice_margin", 0.0)
    slice_margin = cfg.get("slice_margin", 0.5)
    lambda_logit_kd = cfg.get("lambda_logit_kd", 0.0)
    tau_kd = cfg.get("tau_kd", 2.0)
    lambda_feat_kd = cfg.get("lambda_feat_kd", 0.0)
    ewc_lambda = cfg.get("ewc_lambda", 0.0)
    si_lambda = cfg.get("si_lambda", 0.0)

    exp_config = {**cfg, "device": str(DEVICE)}

    with ExperimentManager(
        run_name, exp_config, script_file=script_file or __file__, experiment_group="split_cifar100"
    ) as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = build_model(
            backbone_name=backbone_name,
            num_tasks=NUM_TASKS,
            num_classes_per_task=CLASSES_PER_TASK,
        ).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[cfg["epochs_per_task"] // 2, cfg["epochs_per_task"] * 3 // 4], gamma=0.1
        )

        prev_model = None
        ewc_fisher_star = []
        si_omega_star_list = []

        acc_taw_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
        acc_tag_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

        for task in range(NUM_TASKS):
            logging.info(f"========== Training Task {task} ==========")
            train_loader, _ = task_loaders[task]
            num_classes_so_far = (task + 1) * CLASSES_PER_TASK
            num_old_classes = task * CLASSES_PER_TASK
            need_feat = lambda_feat_kd > 0 or lambda_logit_kd > 0
            if need_feat and prev_model is not None:
                prev_model.eval()

            # SI: 当前任务开始时初始化 omega / theta_prev
            si_omega = {}
            si_theta_prev = {}
            if si_lambda > 0:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        si_omega[n] = torch.zeros_like(p.data, device=DEVICE)
                        si_theta_prev[n] = p.data.clone()

            for ep in range(cfg["epochs_per_task"]):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    if need_feat and prev_model is not None:
                        out = model(x, return_feat=True)
                        logits = out[0] if isinstance(out, tuple) else out
                        feat = out[1] if isinstance(out, tuple) else None
                    else:
                        logits = model(x)
                        feat = None

                    loss = F.cross_entropy(logits[:, :num_classes_so_far], y)

                    if lambda_slice_margin > 0 and task >= 1:
                        loss = loss + lambda_slice_margin * slice_margin_loss(
                            logits[:, :num_classes_so_far], y, task + 1, margin=slice_margin, device=DEVICE
                        )
                    if lambda_logit_kd > 0 and prev_model is not None and num_old_classes > 0:
                        with torch.no_grad():
                            old_logits = prev_model(x)
                        loss = loss + lambda_logit_kd * logit_kd_loss(
                            logits, old_logits, num_old_classes, tau=tau_kd, device=DEVICE
                        )
                    if lambda_feat_kd > 0 and prev_model is not None and feat is not None:
                        with torch.no_grad():
                            _, old_feat = prev_model(x, return_feat=True)
                        loss = loss + lambda_feat_kd * F.mse_loss(feat, old_feat)
                    if ewc_lambda > 0 and ewc_fisher_star:
                        loss = loss + ewc_lambda * ewc_penalty(model, ewc_fisher_star, device=DEVICE)
                    if si_lambda > 0 and si_omega_star_list:
                        loss = loss + si_lambda * si_penalty(model, si_omega_star_list, device=DEVICE)

                    optimizer.zero_grad()
                    loss.backward()
                    if si_lambda > 0 and si_omega:
                        update_si_omega(si_omega, si_theta_prev, model, device=DEVICE)
                    optimizer.step()
                scheduler.step()
                if (ep + 1) % 10 == 0:
                    logging.info(f"  Task {task} Epoch {ep+1}/{cfg['epochs_per_task']} loss={loss.item():.4f}")

            prev_model = copy.deepcopy(model).to(DEVICE)
            freeze_model(prev_model)

            if ewc_lambda > 0:
                fisher = compute_fisher(
                    model, task_loaders, task, num_classes_so_far, device=DEVICE, num_samples=1000
                )
                star = {n: p.data.clone() for n, p in model.named_parameters() if n in fisher}
                ewc_fisher_star.append((fisher, star))
                logging.info(f"EWC: stored Fisher for task {task}")
            if si_lambda > 0 and si_omega:
                omega_copy = {n: t.clone() for n, t in si_omega.items()}
                star_si = {n: p.data.clone() for n, p in model.named_parameters() if n in si_omega}
                si_omega_star_list.append((omega_copy, star_si))
                logging.info(f"SI: stored omega for task {task}")

            # 评估所有已学任务
            acc_taw_row, acc_tag_row = compute_acc_matrices(model, task_loaders, task, DEVICE)
            for u in range(task + 1):
                acc_taw_matrix[task, u] = acc_taw_row[u]
                acc_tag_matrix[task, u] = acc_tag_row[u]

            avg_taw = np.mean(acc_taw_row) * 100
            avg_tag = np.mean(acc_tag_row) * 100
            logging.info(
                f">>> After Task {task}: TAw avg={avg_taw:.2f}% | TAg avg={avg_tag:.2f}% "
                f"(TAw row: {[f'{x*100:.1f}%' for x in acc_taw_row]} | TAg row: {[f'{x*100:.1f}%' for x in acc_tag_row]})"
            )

        # 最终指标（forgetting/bwt 接收 list of lists）
        acc_tag_list = acc_tag_matrix.tolist()
        forget_tag = compute_forgetting(acc_tag_list)
        bwt_tag = compute_bwt(acc_tag_list)
        final_avg_class_il = float(acc_tag_matrix[NUM_TASKS - 1, :].mean())

        logging.info("========== FINAL (TAg / Class-IL) ==========")
        logging.info("Forgetting per task: " + ", ".join([f"T{k}={forget_tag[k]*100:.2f}%" for k in sorted(forget_tag)]))
        logging.info(f"BWT: {bwt_tag*100:.2f}%")
        logging.info(f"Final average Class-IL (TAg): {final_avg_class_il*100:.2f}%")

        metrics = {
            "final_avg_class_il": final_avg_class_il,
            "bwt": float(bwt_tag),
            "forgetting": {int(k): float(v) for k, v in forget_tag.items()},
            "acc_taw_last_row": acc_taw_matrix[NUM_TASKS - 1, :].tolist(),
            "acc_tag_last_row": acc_tag_matrix[NUM_TASKS - 1, :].tolist(),
        }

        ts = save_adagauss_format_results(mgr.exp_dir, acc_taw_matrix, acc_tag_matrix)
        mgr.finish(metrics, save_model=save_model_checkpoint)
        logging.info(f"AdaGauss-format results saved under {mgr.exp_dir}/results/ (timestamp {ts})")
    return metrics
