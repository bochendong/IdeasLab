# ===== Split CIFAR-10 实验基础模块（无 Replay）=====
"""共享训练逻辑、CNN 模型、日志等。与 split_mnist 协议一致：5 任务、Class-IL、无回放。"""
import os
import sys
import copy
import random
import logging
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    _torch_lib = None
    for _d in [
        os.path.join(_PROJECT_ROOT, ".venv", "Lib", "site-packages", "torch", "lib"),
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    ]:
        if os.path.isdir(_d):
            _torch_lib = _d
            break
    if _torch_lib:
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)

import torch
from experiment_manager import ExperimentManager, get_output_dir
import torch.nn as nn
import torch.nn.functional as F

from split_cifar10.common import (
    TASK_DIR,
    build_all_task_loaders,
    cal_acc_class_il,
    evaluate_after_task,
    compute_forgetting,
    compute_bwt,
)

# CIFAR-10 特征维度（CNN backbone 输出）
FEAT_DIM = 256

DEFAULT_CONFIG = {
    "batch_size": 128,
    "epochs_per_task": 10,
    "lr": 1e-3,
    "lambda_slice": 2.0,
    "lambda_feat": 0.5,
    "use_cosine_head": True,
    "seed": 42,
    "freeze_backbone_after_task": -1,
    "ewc_lambda": 0.0,
    "si_lambda": 0.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
OUTPUT_DIR = get_output_dir("split_cifar10")
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
# Model: CNN backbone + Cosine head（与 MNIST CosineMLP 接口一致）
# ------------------------
class CosineCNN(nn.Module):
    """3 层 CNN + 256 维特征 + 10 类 cosine head。输入 (B, 3, 32, 32)。"""
    def __init__(self, out_dim=10, hidden_dim=FEAT_DIM, init_scale=10.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Parameter(torch.empty(out_dim, hidden_dim))
        nn.init.xavier_normal_(self.W)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        h = self.backbone(x)
        feat = self.fc(h)
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
    if hasattr(model, "backbone_model"):
        return get_backbone_params(model.backbone_model)
    if hasattr(model, "backbone"):
        params = list(model.backbone.parameters())
        if hasattr(model, "fc"):
            params += list(model.fc.parameters())
        return params
    return []


def get_head_params(model):
    if hasattr(model, "W"):
        return [model.W, model.logit_scale]
    return []


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
        feat_terms.append(F.mse_loss(student_feat, t_feat) if lambda_feat > 0 else torch.tensor(0.0, device=device))
    return lambda_slice * torch.stack(struct_terms).mean() + lambda_feat * torch.stack(feat_terms).mean()


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


def compute_fisher(model, task_loaders, task_id, valid_out_dim, device=DEVICE, num_samples=1000):
    model.eval()
    fisher = {n: torch.zeros_like(p.data, device=device) for n, p in model.named_parameters() if p.requires_grad}
    counted = 0
    for x, y in task_loaders[task_id][0]:
        if counted >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :valid_out_dim], y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and n in fisher:
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


def si_penalty(model, si_omega_star_list, device=DEVICE, eps=1e-8):
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
# 主训练循环（Baseline / EWC / SI）
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
    ewc_lambda = cfg["ewc_lambda"]
    si_lambda = cfg["si_lambda"]

    exp_config = {
        "device": str(DEVICE), "lr": cfg["lr"], "epochs_per_task": cfg["epochs_per_task"],
        "batch_size": cfg["batch_size"], "lambda_slice": lambda_slice, "lambda_feat": lambda_feat,
        "ewc_lambda": ewc_lambda, "si_lambda": si_lambda, "seed": cfg["seed"],
    }
    with ExperimentManager(run_name, exp_config, script_file=script_file or __file__, experiment_group="split_cifar10") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)
        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = CosineCNN().to(DEVICE)
        prev_models = []
        ewc_fisher_star = []
        si_omega_star_list = []
        task_il_matrix, class_il_matrix = [], []

        for task in range(5):
            logging.info(f"========== Training Task {task}: classes {TASK_DIR[task]} ==========")
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)

            si_omega = {}
            si_theta_prev = {}
            if si_lambda > 0:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        si_omega[n] = torch.zeros_like(p.data, device=DEVICE)
                        si_theta_prev[n] = p.data.clone()

            step = 0
            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()
                    logits, feat = model(x)
                    c_loss = criterion(logits[:, :valid_out_dim], y)
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    ewc_loss = ewc_lambda * ewc_penalty(model, ewc_fisher_star, device=DEVICE) if ewc_lambda > 0 and ewc_fisher_star else torch.tensor(0.0, device=DEVICE)
                    si_loss = si_lambda * si_penalty(model, si_omega_star_list, device=DEVICE) if si_lambda > 0 and si_omega_star_list else torch.tensor(0.0, device=DEVICE)
                    loss = c_loss + r_loss + ewc_loss + si_loss
                    optimizer.zero_grad()
                    loss.backward()
                    if si_lambda > 0 and si_omega:
                        update_si_omega(si_omega, si_theta_prev, model, device=DEVICE)
                    optimizer.step()

                    if step % 50 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] loss={loss.item():.4f} c={c_loss.item():.4f} r={r_loss.item():.4f}"]
                        if ewc_loss.item() > 0:
                            msg.append(f"ewc={ewc_loss.item():.4f}")
                        if si_loss.item() > 0:
                            msg.append(f"si={si_loss.item():.4f}")
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(model, task_loaders[t][0], valid_out_dim, device=DEVICE)
                            msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                        logging.info(" | ".join(msg))
                    step += 1

            teacher = copy.deepcopy(model).to(DEVICE)
            teacher = freeze_model(teacher)
            prev_models.append(teacher)

            if ewc_lambda > 0:
                fisher = compute_fisher(model, task_loaders, task, valid_out_dim, device=DEVICE, num_samples=1000)
                star = {n: p.data.clone() for n, p in model.named_parameters() if n in fisher}
                ewc_fisher_star.append((fisher, star))
            if si_lambda > 0 and si_omega:
                si_omega_star_list.append(({n: t.clone() for n, t in si_omega.items()}, {n: p.data.clone() for n, p in model.named_parameters() if n in si_omega}))

            task_il, class_il, avg_task, avg_class = evaluate_after_task(model, task, task_loaders, device=DEVICE)
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

    print("Done. Log:", log_path)
    return log_path
