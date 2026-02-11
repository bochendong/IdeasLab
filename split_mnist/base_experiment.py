# ===== Split MNIST 实验基础模块（无 Replay）=====
"""共享训练逻辑、模型定义、日志等。各实验通过 config 覆盖参数。"""
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
from experiment_manager import ExperimentManager, get_output_dir
import torch.nn as nn
import torch.nn.functional as F

from split_mnist.common import (
    TASK_DIR,
    build_all_task_loaders,
    cal_acc_class_il,
    evaluate_after_task,
    compute_forgetting,
    compute_bwt,
)

# 默认配置（可被实验覆盖）
DEFAULT_CONFIG = {
    "batch_size": 128,
    "epochs_per_task": 4,
    "lr": 5e-3,
    "lambda_slice": 2.0,
    "lambda_feat": 0.5,
    "use_cosine_head": True,
    "seed": 42,
    "freeze_backbone_after_task": -1,  # -1 = 从不冻结
    "ewc_lambda": 0.0,  # 0 = 不使用 EWC
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
OUTPUT_DIR = get_output_dir("split_mnist")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# Model
# ------------------------
class VanillaMLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=400):
        super().__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits, feat


class CosineMLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=400, init_scale=10.0):
        super().__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Parameter(torch.empty(out_dim, hidden_dim))
        nn.init.xavier_normal_(self.W)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        feat = self.backbone(x)
        feat_n = F.normalize(feat, dim=1)
        W_n = F.normalize(self.W, dim=1)
        logits = self.logit_scale * (feat_n @ W_n.t())
        return logits, feat


def freeze_model(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def get_backbone_params(model):
    """获取 backbone 参数（用于 EWC / 冻结）"""
    if hasattr(model, "backbone"):
        return list(model.backbone.parameters())
    return []


def get_head_params(model):
    """获取 head 参数"""
    if hasattr(model, "W"):
        return [model.W, model.logit_scale]
    return list(model.head.parameters())


# ------------------------
# Group differences logging
# ------------------------
@torch.no_grad()
def log_group_differences(model, task_done, task_loaders, device=DEVICE):
    model.eval()
    num_tasks = task_done + 1
    valid_out_dim = 2 * num_tasks

    logging.info("========== GROUP DIFFERENCES ==========")
    logging.info(f"Seen tasks: 0..{task_done} | valid_out_dim={valid_out_dim}")

    for g in range(num_tasks):
        test_loader = task_loaders[g][1]
        total = 0

        corr_slice_sum = 0.0
        best_other_sum = 0.0
        gap_sum = 0.0

        win_counts = np.zeros(num_tasks, dtype=np.int64)
        class_il_correct = 0

        slice_strength_sum = np.zeros(num_tasks, dtype=np.float64)

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            logits = logits[:, :valid_out_dim]

            pred = logits.argmax(dim=1)
            class_il_correct += int((pred == y).sum().item())

            win_task = (pred // 2).detach().cpu().numpy()
            for t_id in win_task:
                win_counts[t_id] += 1

            corr_slice = logits[:, 2*g:2*g+2]
            corr_max = corr_slice.max(dim=1).values

            if num_tasks == 1:
                gap = torch.full_like(corr_max, float("nan"))
            else:
                other_max_per_slice = []
                for j in range(num_tasks):
                    if j == g:
                        continue
                    s = logits[:, 2*j:2*j+2].max(dim=1).values
                    other_max_per_slice.append(s.unsqueeze(1))
                other_max = torch.cat(other_max_per_slice, dim=1).max(dim=1).values
                gap = corr_max - other_max
                best_other_sum += float(other_max.sum().item())
                gap_sum += float(gap[~torch.isnan(gap)].sum().item())

            corr_slice_sum += float(corr_max.sum().item())
            for j in range(num_tasks):
                smax = logits[:, 2*j:2*j+2].max(dim=1).values
                slice_strength_sum[j] += float(smax.sum().item())

            total += x.size(0)

        total = max(total, 1)
        corr_slice_avg = corr_slice_sum / total
        ci_acc = class_il_correct / total

        if num_tasks == 1:
            best_other_avg = float("nan")
            gap_avg = float("nan")
        else:
            best_other_avg = best_other_sum / total
            gap_avg = gap_sum / total

        win_rate = win_counts / total
        win_str = " | ".join([f"W{t}={win_rate[t]*100:.1f}%" for t in range(num_tasks)])
        strength_avg = slice_strength_sum / total
        strength_str = " | ".join([f"S{t}={strength_avg[t]:.2f}" for t in range(num_tasks)])

        logging.info(
            f"[Group=Task{g}] CI_acc={ci_acc*100:.2f}% | "
            f"corr_slice_max={corr_slice_avg:.3f} | best_other_max={best_other_avg:.3f} | gap={gap_avg:.3f}"
        )
        logging.info(f"[Group=Task{g}] winner_task_dist: {win_str}")
        logging.info(f"[Group=Task{g}] slice_strength : {strength_str}")

    logging.info("======================================")


# ------------------------
# EWC: 计算 Fisher 信息矩阵（对角近似）
# ------------------------
def compute_fisher(model, task_loaders, task_id, valid_out_dim, device=DEVICE, num_samples=1000):
    """在 task_id 的 train 数据上计算 Fisher 对角（仅 backbone 参数）"""
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
        logits, _ = model(x)
        logits = logits[:, :valid_out_dim]
        loss = F.cross_entropy(logits, y)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        counted += x.size(0)

    for n in fisher:
        fisher[n] /= max(counted, 1)
    return fisher


def ewc_penalty(model, ewc_fisher_star, device=DEVICE):
    """EWC 惩罚项：sum_k Fisher_k * (theta - theta_star_k)^2"""
    loss = torch.tensor(0.0, device=device)
    for fisher, star in ewc_fisher_star:
        for n, p in model.named_parameters():
            if n in fisher and p.requires_grad:
                loss = loss + (fisher[n] * (p - star[n]) ** 2).sum()
    return loss


# ------------------------
# Regularization (slice drift + feature drift)
# ------------------------
def reg_loss_slice(student_logits, student_feat, x, prev_models, task_num, lambda_slice, lambda_feat, device=DEVICE):
    if task_num == 0 or len(prev_models) == 0:
        return torch.tensor(0.0, device=device)

    struct_terms = []
    feat_terms = []

    for k, teacher in enumerate(prev_models):
        with torch.no_grad():
            t_logits, t_feat = teacher(x)

        s_slice = student_logits[:, 2*k:2*k+2]
        t_slice = t_logits[:, 2*k:2*k+2]
        struct_terms.append((t_slice - s_slice).abs().mean())

        if lambda_feat > 0:
            feat_terms.append(F.mse_loss(student_feat, t_feat))
        else:
            feat_terms.append(torch.tensor(0.0, device=device))

    struct = torch.stack(struct_terms).mean()
    feat = torch.stack(feat_terms).mean()
    return lambda_slice * struct + lambda_feat * feat


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

    task_loaders = build_all_task_loaders(DATA_DIR, cfg["batch_size"])
    lambda_slice = cfg["lambda_slice"]
    lambda_feat = cfg["lambda_feat"]
    freeze_after = cfg["freeze_backbone_after_task"]
    ewc_lambda = cfg["ewc_lambda"]

    exp_config = {
        "device": str(DEVICE),
        "lr": cfg["lr"],
        "epochs_per_task": cfg["epochs_per_task"],
        "batch_size": cfg["batch_size"],
        "lambda_slice": lambda_slice,
        "lambda_feat": lambda_feat,
        "use_cosine_head": cfg["use_cosine_head"],
        "seed": cfg["seed"],
        "freeze_backbone_after_task": freeze_after,
        "ewc_lambda": ewc_lambda,
    }

    with ExperimentManager(
        run_name, exp_config, script_file=script_file or __file__, experiment_group="split_mnist"
    ) as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        ModelCls = CosineMLP if cfg["use_cosine_head"] else VanillaMLP
        model = ModelCls().to(DEVICE)

        prev_models = []
        ewc_fisher_star = []  # [(fisher_dict, star_dict), ...]

        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Training Task {task}: classes {TASK_DIR[task]} ==========")

            # 冻结 backbone（若配置）
            if freeze_after >= 0 and task > freeze_after:
                for p in get_backbone_params(model):
                    p.requires_grad = False
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])
                logging.info(f"Backbone frozen (task > {freeze_after}), only training head")
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)

            step = 0
            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()

                    logits, feat = model(x)
                    c_loss = criterion(logits[:, :valid_out_dim], y)
                    r_loss = reg_loss_slice(
                        logits, feat, x, prev_models, task,
                        lambda_slice, lambda_feat, device=DEVICE
                    )

                    ewc_loss = torch.tensor(0.0, device=DEVICE)
                    if ewc_lambda > 0 and ewc_fisher_star:
                        ewc_loss = ewc_lambda * ewc_penalty(model, ewc_fisher_star, device=DEVICE)

                    loss = c_loss + r_loss + ewc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 25 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] "
                               f"loss={loss.item():.4f} c={c_loss.item():.4f} r={r_loss.item():.4f}"]
                        if ewc_loss.item() > 0:
                            msg.append(f"ewc={ewc_loss.item():.4f}")
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(
                                model, task_loaders[t][0], valid_out_dim, device=DEVICE
                            )
                            msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                        logging.info(" | ".join(msg))

                    step += 1

            teacher = copy.deepcopy(model).to(DEVICE)
            teacher = freeze_model(teacher)
            prev_models.append(teacher)

            # EWC: 计算当前 task 的 Fisher 并保存
            if ewc_lambda > 0:
                fisher = compute_fisher(
                    model, task_loaders, task, valid_out_dim,
                    device=DEVICE, num_samples=1000
                )
                star = {n: p.data.clone() for n, p in model.named_parameters() if n in fisher}
                ewc_fisher_star.append((fisher, star))
                logging.info(f"EWC: stored Fisher for task {task}")

            task_il, class_il, avg_task, avg_class = evaluate_after_task(
                model, task, task_loaders, device=DEVICE
            )
            task_il_matrix.append(task_il)
            class_il_matrix.append(class_il)

            msg_ti = " | ".join([f"T{k}={task_il[k]*100:.2f}%" for k in range(task + 1)])
            msg_ci = " | ".join([f"T{k}={class_il[k]*100:.2f}%" for k in range(task + 1)])
            logging.info(f"[After Task {task}] Task-IL : {msg_ti} | Avg={avg_task*100:.2f}%")
            logging.info(f"[After Task {task}] Class-IL: {msg_ci} | Avg={avg_class*100:.2f}%")

            log_group_differences(model, task, task_loaders)

        forget = compute_forgetting(class_il_matrix)
        bwt = compute_bwt(class_il_matrix)
        final_avg_class = float(np.mean(list(class_il_matrix[-1].values())))

        logging.info("========== FINAL METRICS (Class-IL) ==========")
        logging.info("Forgetting per task: " + ", ".join([f"T{k}={forget[k]*100:.2f}%" for k in sorted(forget)]))
        logging.info(f"BWT: {bwt*100:.2f}%")
        logging.info(f"Final average Class-IL accuracy: {final_avg_class*100:.2f}%")

        metrics = {
            "final_avg_class_il": float(final_avg_class),
            "bwt": float(bwt),
            "forgetting": {int(k): float(v) for k, v in forget.items()},
            "task_il_per_task": {t: {int(k): float(v) for k, v in task_il_matrix[t].items()} for t in range(5)},
            "class_il_per_task": {t: {int(k): float(v) for k, v in class_il_matrix[t].items()} for t in range(5)},
        }

        mgr.finish(metrics, model=model, save_model_checkpoint=save_model_checkpoint)

    print("Done.")
    print("Log:", log_path)
    return log_path
