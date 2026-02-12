# ===== Exp22: Slice 高斯化 + Anti-Collapse (AdaGauss 风) =====
"""每类存均值+对角协方差，用 N(μ,Σ) 采样伪特征；Anti-Collapse 约束协方差正定，缓解 task-recency bias。"""
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
    DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, CosineMLP, freeze_model,
    reg_loss_slice, log_group_differences,
)


def compute_prototypes_and_cov(model, task_loaders, task_id, device, eps=1e-6):
    """计算每类均值与对角协方差。"""
    model.eval()
    feat_by_class = {c: [] for c in TASK_DIR[task_id]}
    with torch.no_grad():
        for x, y in task_loaders[task_id][0]:
            x = x.to(device)
            _, feat = model(x)
            for i in range(x.size(0)):
                c = int(y[i].item())
                feat_by_class[c].append(feat[i])
    protos = {}
    covs = {}
    for c in TASK_DIR[task_id]:
        if feat_by_class[c]:
            stacked = torch.stack(feat_by_class[c])
            protos[c] = stacked.mean(dim=0)
            diff = stacked - protos[c]
            covs[c] = (diff * diff).mean(dim=0) + eps
        else:
            protos[c] = torch.zeros(400, device=device)
            covs[c] = torch.ones(400, device=device)
    return protos, covs


def sample_gaussian_prototypes(protos_dict, covs_dict, n_per_class, device):
    """从 N(μ, diag(σ²)) 采样。"""
    feats, labels = [], []
    for c, proto in protos_dict.items():
        cov = covs_dict.get(c, torch.ones_like(proto, device=device))
        std = torch.sqrt(torch.clamp(cov, min=1e-8))
        for _ in range(n_per_class):
            z = torch.randn_like(proto, device=device)
            aug = proto + std * z
            feats.append(aug)
            labels.append(c)
    if not feats:
        return None, None
    return torch.stack(feats), torch.tensor(labels, device=device, dtype=torch.long)


def anti_collapse_loss(feat, device, min_std=1.0):
    """Anti-Collapse：约束特征各维度标准差 >= min_std，缓解维度坍缩。"""
    if feat.size(0) < 2:
        return torch.tensor(0.0, device=device)
    std = feat.std(dim=0) + 1e-8
    return F.relu(min_std - std).pow(2).mean()


def run_experiment_slice_gauss_anticollapse(
    run_name: str = "exp22_slice_gauss_anticollapse",
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
    proto_aug_n_per_class: int = 8,
    lambda_ac: float = 0.1,
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
        "proto_aug_n_per_class": proto_aug_n_per_class,
        "lambda_ac": lambda_ac,
    }

    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_mnist") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = CosineMLP().to(DEVICE)
        prev_models = []
        all_protos = {}
        all_covs = {}
        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")
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
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    loss = c_loss + r_loss

                    ac_loss = lambda_ac * anti_collapse_loss(feat, DEVICE)
                    loss = loss + ac_loss

                    if task > 0 and all_protos:
                        old_protos = {c: p for c, p in all_protos.items() if c not in TASK_DIR[task]}
                        old_covs = {c: all_covs.get(c, torch.ones(400, device=DEVICE)) for c in old_protos}
                        if old_protos:
                            feat_aug, label_aug = sample_gaussian_prototypes(old_protos, old_covs, proto_aug_n_per_class, DEVICE)
                            if feat_aug is not None:
                                feat_n = F.normalize(feat_aug, dim=1)
                                W_n = F.normalize(model.W.data[:valid_out_dim].t(), dim=1)
                                logits_aug = model.logit_scale * (feat_n @ W_n)
                                loss = loss + 0.5 * criterion(logits_aug, label_aug)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 25 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] loss={loss.item():.4f} c={c_loss.item():.4f} r={r_loss.item():.4f} ac={ac_loss.item():.4f}"]
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(model, task_loaders[t][0], valid_out_dim, device=DEVICE)
                            msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                        logging.info(" | ".join(msg))
                    step += 1

            protos_t, covs_t = compute_prototypes_and_cov(model, task_loaders, task, DEVICE)
            for c, p in protos_t.items():
                all_protos[c] = p.detach().clone()
            for c, co in covs_t.items():
                all_covs[c] = co.detach().clone()

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
    run_experiment_slice_gauss_anticollapse(
        run_name="exp22_slice_gauss_anticollapse",
        config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__),
        proto_aug_n_per_class=8,
        lambda_ac=0.1,
    )
