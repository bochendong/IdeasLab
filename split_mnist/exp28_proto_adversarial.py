# ===== Exp28: 原型 + 对抗（无 VAE）=====
"""只存特征原型；判别器区分当前 batch 特征 vs 原型+噪声伪旧特征；projector G 将伪旧特征映射为骗过 D，并对 G(伪旧) 做 head CE。"""
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
from split_mnist.exp19_proto_aug_si import compute_prototypes, sample_augmented_prototypes


class Discriminator(nn.Module):
    """判别器：feat -> 0=旧类, 1=新类。"""
    def __init__(self, feat_dim=400, hidden=128):
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


class ProjectorG(nn.Module):
    """将伪旧特征映射到「骗过 D」的空间，保持可分类。"""
    def __init__(self, feat_dim=400, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, feat):
        return self.net(feat)


def run_experiment_proto_adversarial(
    run_name: str = "exp28_proto_adversarial",
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
    proto_aug_r: float = 0.4,
    proto_aug_n_per_class: int = 8,
    lambda_adv: float = 0.1,
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
        "proto_aug_r": proto_aug_r,
        "proto_aug_n_per_class": proto_aug_n_per_class,
        "lambda_adv": lambda_adv,
    }

    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_mnist") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)

        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = CosineMLP().to(DEVICE)
        discriminator = Discriminator(feat_dim=400).to(DEVICE)
        projector_g = ProjectorG(feat_dim=400).to(DEVICE)
        prev_models = []
        all_protos = {}
        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr"] * 0.5)
            opt_g = torch.optim.Adam(projector_g.parameters(), lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)
            step = 0

            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()
                    discriminator.train()
                    projector_g.train()

                    logits, feat = model(x)
                    c_loss = criterion(logits[:, :valid_out_dim], y)
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    loss = c_loss + r_loss

                    feat_new = feat.detach()
                    if task > 0 and all_protos:
                        old_protos = {c: p for c, p in all_protos.items() if c not in TASK_DIR[task]}
                        if old_protos:
                            feat_aug, label_aug = sample_augmented_prototypes(old_protos, proto_aug_n_per_class, proto_aug_r, DEVICE)
                            if feat_aug is not None:
                                feat_aug = feat_aug.to(DEVICE)
                                feat_old_proj = projector_g(feat_aug)

                                d_out_new = discriminator(feat_new)
                                d_out_old = discriminator(feat_old_proj.detach())
                                loss_d = F.binary_cross_entropy_with_logits(d_out_new, torch.ones_like(d_out_new)) + \
                                         F.binary_cross_entropy_with_logits(d_out_old, torch.zeros_like(d_out_old))
                                opt_d.zero_grad()
                                loss_d.backward()
                                opt_d.step()

                                d_out_old_for_g = discriminator(feat_old_proj)
                                loss_adv_g = F.binary_cross_entropy_with_logits(d_out_old_for_g, torch.ones_like(d_out_old_for_g))
                                feat_old_n = F.normalize(feat_old_proj, dim=1)
                                W_n = F.normalize(model.W.data[:valid_out_dim].t(), dim=1)
                                logits_aug = model.logit_scale * (feat_old_n @ W_n)
                                loss_ce_old = criterion(logits_aug, label_aug)
                                loss_g = lambda_adv * loss_adv_g + 0.5 * loss_ce_old
                                opt_g.zero_grad()
                                loss_g.backward()
                                opt_g.step()
                                # 不在主 loss 中再次加入 loss_ce_old，避免对同一计算图 backward 两次

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

            protos_t = compute_prototypes(model, task_loaders, task, DEVICE)
            for c, p in protos_t.items():
                all_protos[c] = p.detach().clone()

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
    run_experiment_proto_adversarial(
        run_name="exp28_proto_adversarial",
        config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__),
        proto_aug_r=0.4,
        proto_aug_n_per_class=8,
        lambda_adv=0.1,
    )
