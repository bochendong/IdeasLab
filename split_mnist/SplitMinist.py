# ===== Colab-ready Split MNIST Continual Learning (NO REPLAY) + Group-Difference Logging (FIXED) =====
import os
import sys
import copy
import random
import logging
import numpy as np

# 项目根目录（split_mnist 的父目录），用于导入和路径
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

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 128
EPOCHS_PER_TASK = 4
LR = 5e-3
OUTPUT_DIR = get_output_dir("split_mnist")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
SEED = 42

LAMBDA_SLICE = 2.0
LAMBDA_FEAT  = 0.5
USE_COSINE_HEAD = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_LOADERS = build_all_task_loaders(DATA_DIR, BATCH_SIZE)

# ------------------------
# Seed + logging
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def setup_logging(file_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(file_name), logging.StreamHandler()],
        force=True,
    )

# ------------------------
# Group differences logging (FIXED for task_done=0)
# ------------------------
@torch.no_grad()
def log_group_differences(model, task_done, device=DEVICE):
    model.eval()
    num_tasks = task_done + 1
    valid_out_dim = 2 * num_tasks

    logging.info("========== GROUP DIFFERENCES ==========")
    logging.info(f"Seen tasks: 0..{task_done} | valid_out_dim={valid_out_dim}")

    for g in range(num_tasks):
        test_loader = TASK_LOADERS[g][1]
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

            # if only one task exists, there is no "other slice"
            if num_tasks == 1:
                other_max = torch.zeros_like(corr_max)  # dummy, not used
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

            corr_slice_sum += float(corr_max.sum().item())
            if num_tasks > 1:
                best_other_sum += float(other_max.sum().item())
                gap_sum += float(gap[~torch.isnan(gap)].sum().item())

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

# ------------------------
# Teacher snapshots
# ------------------------
def freeze_model(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

# ------------------------
# Regularization (slice drift + feature drift)
# ------------------------
def reg_loss_slice(student_logits, student_feat, x, prev_models, task_num):
    if task_num == 0 or len(prev_models) == 0:
        return torch.tensor(0.0, device=DEVICE)

    struct_terms = []
    feat_terms = []

    for k, teacher in enumerate(prev_models):
        with torch.no_grad():
            t_logits, t_feat = teacher(x)

        s_slice = student_logits[:, 2*k:2*k+2]
        t_slice = t_logits[:, 2*k:2*k+2]
        struct_terms.append((t_slice - s_slice).abs().mean())

        if LAMBDA_FEAT > 0:
            feat_terms.append(F.mse_loss(student_feat, t_feat))
        else:
            feat_terms.append(torch.tensor(0.0, device=DEVICE))

    struct = torch.stack(struct_terms).mean()
    feat = torch.stack(feat_terms).mean()
    return LAMBDA_SLICE * struct + LAMBDA_FEAT * feat

# ------------------------
# Train one task
# ------------------------
def train_one_task(model, task_num, optimizer, criterion, prev_models, epochs=EPOCHS_PER_TASK, log_every=25):
    train_loader, _ = TASK_LOADERS[task_num]
    valid_out_dim = 2 * (task_num + 1)

    step = 0
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()

            logits, feat = model(x)
            c_loss = criterion(logits[:, :valid_out_dim], y)
            r_loss = reg_loss_slice(logits, feat, x, prev_models, task_num)
            loss = c_loss + r_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                msg = [f"[Task {task_num} | ep {ep} | step {step}] "
                       f"loss={loss.item():.4f} c={c_loss.item():.4f} r={r_loss.item():.4f}"]
                for t in range(task_num + 1):
                    acc_ci = cal_acc_class_il(
                        model, TASK_LOADERS[t][0], valid_out_dim, device=DEVICE
                    )
                    msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                logging.info(" | ".join(msg))

            step += 1

# ------------------------
# Full run
# ------------------------
def _build_config():
    return {
        "device": str(DEVICE),
        "lr": LR,
        "epochs_per_task": EPOCHS_PER_TASK,
        "batch_size": BATCH_SIZE,
        "lambda_slice": LAMBDA_SLICE,
        "lambda_feat": LAMBDA_FEAT,
        "use_cosine_head": USE_COSINE_HEAD,
        "seed": SEED,
    }


def train_split_mnist(run_name="split_mnist_groupdiff_no_replay", save_model_checkpoint=False):
    config = _build_config()

    with ExperimentManager(
        run_name, config, script_file=__file__, experiment_group="split_mnist"
    ) as mgr:
        # 日志直接写入实验目录
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"LR={LR}, EPOCHS_PER_TASK={EPOCHS_PER_TASK}, BATCH_SIZE={BATCH_SIZE}")
        logging.info(f"LAMBDA_SLICE={LAMBDA_SLICE}, LAMBDA_FEAT={LAMBDA_FEAT}, USE_COSINE_HEAD={USE_COSINE_HEAD}")

        model = (CosineMLP() if USE_COSINE_HEAD else VanillaMLP()).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        prev_models = []
        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Training Task {task}: classes {TASK_DIR[task]} ==========")
            train_one_task(model, task, optimizer, criterion, prev_models, epochs=EPOCHS_PER_TASK, log_every=25)

            teacher = copy.deepcopy(model).to(DEVICE)
            teacher = freeze_model(teacher)
            prev_models.append(teacher)

            task_il, class_il, avg_task, avg_class = evaluate_after_task(
                model, task, TASK_LOADERS, device=DEVICE
            )
            task_il_matrix.append(task_il)
            class_il_matrix.append(class_il)

            msg_ti = " | ".join([f"T{k}={task_il[k]*100:.2f}%" for k in range(task + 1)])
            msg_ci = " | ".join([f"T{k}={class_il[k]*100:.2f}%" for k in range(task + 1)])
            logging.info(f"[After Task {task}] Task-IL : {msg_ti} | Avg={avg_task*100:.2f}%")
            logging.info(f"[After Task {task}] Class-IL: {msg_ci} | Avg={avg_class*100:.2f}%")

            log_group_differences(model, task)

        forget = compute_forgetting(class_il_matrix)
        bwt = compute_bwt(class_il_matrix)
        final_avg_class = float(np.mean(list(class_il_matrix[-1].values())))

        logging.info("========== FINAL METRICS (Class-IL) ==========")
        logging.info("Forgetting per task: " + ", ".join([f"T{k}={forget[k]*100:.2f}%" for k in sorted(forget)]))
        logging.info(f"BWT: {bwt*100:.2f}%")
        logging.info(f"Final average Class-IL accuracy: {final_avg_class*100:.2f}%")

        # 整理指标（转为 JSON 可序列化）
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

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    train_split_mnist("split_mnist_groupdiff_no_replay_fixed")
