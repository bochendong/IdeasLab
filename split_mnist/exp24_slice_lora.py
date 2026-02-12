# ===== Exp24: SD-LoRA 式 (Slice-level LoRA) =====
"""每个 slice 配低秩 LoRA：W_slice = W_base + α * A @ B；旧任务 A,B 冻结，只更新 α 或新任务。"""
import os
import sys
import copy
import logging
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    _tl = None
    for _d in [os.path.join(_PROJECT_ROOT, ".venv", "Lib", "site-packages", "torch", "lib"), os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")]:
        if os.path.isdir(_d):
            _tl = _d
            break
    if _tl:
        os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_tl)

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment_manager import ExperimentManager, get_output_dir

from split_mnist.common import TASK_DIR, build_all_task_loaders, cal_acc_class_il, evaluate_after_task, compute_forgetting, compute_bwt
from split_mnist.base_experiment import (
    DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, freeze_model,
    reg_loss_slice, log_group_differences,
)


class CosineMLPWithSliceLoRA(nn.Module):
    """Cosine head + 每 slice 一个 LoRA：W_t = W_base_t + alpha_t * A_t @ B_t。"""
    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=400, init_scale=10.0, lora_rank=8):
        super().__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.num_slices = out_dim // 2
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.W_base = nn.Parameter(torch.empty(out_dim, hidden_dim))
        nn.init.xavier_normal_(self.W_base)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.lora_alpha = nn.ParameterList()
        self._lora_initialized = 0

    def add_lora_slice(self):
        r = self.lora_rank
        h = self.hidden_dim
        A = nn.Parameter(torch.randn(2, r) * 0.01)
        B = nn.Parameter(torch.randn(r, h) * 0.01)
        alpha = nn.Parameter(torch.tensor(1.0))
        self.lora_A.append(A)
        self.lora_B.append(B)
        self.lora_alpha.append(alpha)
        self._lora_initialized += 1

    def get_W_for_tasks(self, num_tasks):
        """返回 (2*num_tasks, hidden_dim) 的权重。"""
        W_list = []
        for t in range(num_tasks):
            w_base = self.W_base[2*t:2*t+2]
            if t < len(self.lora_A):
                A, B = self.lora_A[t], self.lora_B[t]
                alpha = self.lora_alpha[t]
                w_delta = alpha * (A @ B)
                W_list.append(w_base + w_delta)
            else:
                W_list.append(w_base)
        return torch.cat(W_list, dim=0)

    def forward(self, x, num_tasks=None):
        x = x.view(-1, self.in_dim)
        feat = self.backbone(x)
        feat_n = F.normalize(feat, dim=1)
        if num_tasks is None:
            num_tasks = max(1, len(self.lora_A) if self.lora_A else 1)
        W = self.get_W_for_tasks(num_tasks)
        W_n = F.normalize(W.t(), dim=1)
        logits = self.logit_scale * (feat_n @ W_n)
        return logits, feat


def run_experiment_slice_lora(
    run_name: str = "exp24_slice_lora",
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
    lora_rank: int = 8,
):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    task_loaders = build_all_task_loaders(DATA_DIR, cfg["batch_size"])
    lambda_slice = cfg["lambda_slice"]
    lambda_feat = cfg["lambda_feat"]

    exp_config = {
        "device": str(DEVICE),
        "lr": cfg["lr"],
        "epochs_per_task": cfg["epochs_per_task"],
        "batch_size": cfg["batch_size"],
        "lambda_slice": lambda_slice,
        "lambda_feat": lambda_feat,
        "lora_rank": lora_rank,
    }

    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_mnist") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = CosineMLPWithSliceLoRA(lora_rank=lora_rank).to(DEVICE)
        prev_models = []
        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")
            model.add_lora_slice()

            if task > 0:
                for t in range(task):
                    model.lora_A[t].requires_grad = False
                    model.lora_B[t].requires_grad = False

            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)
            step = 0

            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()

                    logits, feat = model(x, num_tasks=task + 1)
                    logits = logits[:, :valid_out_dim]
                    c_loss = criterion(logits, y)
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    loss = c_loss + r_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 25 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] loss={loss.item():.4f}"]
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(model, task_loaders[t][0], valid_out_dim, device=DEVICE)
                            msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                        logging.info(" | ".join(msg))
                    step += 1

            prev_models.append(freeze_model(copy.deepcopy(model).to(DEVICE)))

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

    print("Done.")
    print("Log:", log_path)
    return log_path


if __name__ == "__main__":
    run_experiment_slice_lora(
        run_name="exp24_slice_lora",
        config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__),
        lora_rank=8,
    )
