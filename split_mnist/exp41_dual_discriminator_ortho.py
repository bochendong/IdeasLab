# ===== Exp41: 双判别器 + 新类特征与旧原型正交（PRL PGRU 风）=====
"""Exp32 + 新任务样本特征与旧类原型正交化损失，新类推入预留空间。"""
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
from experiment_manager import ExperimentManager

from split_mnist.common import TASK_DIR, build_all_task_loaders, cal_acc_class_il, evaluate_after_task, compute_forgetting, compute_bwt
from split_mnist.base_experiment import DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, CosineMLP, freeze_model, reg_loss_slice, log_group_differences
from split_mnist.exp9_vae_pseudo_replay import CVAE, train_cvae, sample_from_vaes
from split_mnist.exp32_dual_discriminator import FeatDiscriminator, SliceDiscriminator, slice_max_vector, NUM_TASKS


def compute_prototypes(model, task_loaders, task_id, device):
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
    for c in TASK_DIR[task_id]:
        if feat_by_class[c]:
            protos[c] = torch.stack(feat_by_class[c]).mean(dim=0)
        else:
            protos[c] = torch.zeros(400, device=device)
    return protos


def run_experiment(run_name="exp41_dual_discriminator_ortho", config=None, save_model_checkpoint=False, script_file=None,
    vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_ort=0.1):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    task_loaders = build_all_task_loaders(DATA_DIR, cfg["batch_size"])
    lambda_slice, lambda_feat = cfg["lambda_slice"], cfg["lambda_feat"]

    exp_config = {"device": str(DEVICE), "lr": cfg["lr"], "epochs_per_task": cfg["epochs_per_task"],
        "batch_size": cfg["batch_size"], "lambda_slice": lambda_slice, "lambda_feat": lambda_feat,
        "vae_epochs": vae_epochs, "vae_n_fake_per_task": vae_n_fake_per_task,
        "lambda_adv_feat": lambda_adv_feat, "lambda_adv_slice": lambda_adv_slice, "lambda_ort": lambda_ort}

    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_mnist") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)
        logging.info(f"Device: {DEVICE}")
        logging.info(f"Config: {exp_config}")

        model = CosineMLP().to(DEVICE)
        feat_disc = FeatDiscriminator(feat_dim=400).to(DEVICE)
        slice_disc = SliceDiscriminator(input_dim=NUM_TASKS).to(DEVICE)
        prev_models, vaes = [], []
        protos_by_task = []
        task_il_matrix, class_il_matrix = [], []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")
            vae = CVAE(in_dim=784, num_classes=2, latent_dim=32, hidden=256).to(DEVICE)
            vae = train_cvae(vae, task_loaders[task][0], DEVICE, epochs=vae_epochs)
            vaes.append(copy.deepcopy(vae))

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            opt_d_feat = torch.optim.Adam(feat_disc.parameters(), lr=cfg["lr"] * 0.5)
            opt_d_slice = torch.optim.Adam(slice_disc.parameters(), lr=cfg["lr"] * 0.5)
            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)
            num_tasks_cur = task + 1
            step = 0

            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()
                    feat_disc.train()
                    slice_disc.train()

                    logits, feat = model(x)
                    c_loss = criterion(logits[:, :valid_out_dim], y)
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    loss = c_loss + r_loss

                    if task > 0 and protos_by_task:
                        feat_new_n = F.normalize(feat, dim=1)
                        P_list = [p for protos in protos_by_task for p in protos.values()]
                        P_old = torch.stack(P_list).to(DEVICE)
                        P_old_n = F.normalize(P_old, dim=1)
                        sim = feat_new_n @ P_old_n.t()
                        loss = loss + lambda_ort * (sim.pow(2).mean())

                    feat_new = feat.detach()
                    slice_max_new = slice_max_vector(logits, num_tasks_cur, DEVICE)

                    if task > 0 and vaes:
                        n_fake = min(vae_n_fake_per_task * task, 128)
                        x_fake, y_fake = sample_from_vaes(vaes[:-1], list(range(task)), max(1, n_fake // task), DEVICE)
                        x_fake = x_fake.view(-1, 1, 28, 28)
                        if x_fake.size(0) > 0:
                            _, feat_old = model(x_fake)
                            feat_old = feat_old.detach()
                            logits_fake, _ = model(x_fake)
                            logits_fake = logits_fake[:, :valid_out_dim]
                            num_tasks_old = task
                            slice_max_old = slice_max_vector(logits_fake, num_tasks_old, DEVICE)

                            d_feat_new = feat_disc(feat_new)
                            d_feat_old = feat_disc(feat_old)
                            loss_d_feat = F.binary_cross_entropy_with_logits(d_feat_new, torch.ones_like(d_feat_new)) + F.binary_cross_entropy_with_logits(d_feat_old, torch.zeros_like(d_feat_old))
                            opt_d_feat.zero_grad()
                            loss_d_feat.backward()
                            opt_d_feat.step()

                            d_slice_new = slice_disc(slice_max_new.detach())
                            d_slice_old = slice_disc(slice_max_old.detach())
                            loss_d_slice = F.binary_cross_entropy_with_logits(d_slice_new, torch.ones_like(d_slice_new)) + F.binary_cross_entropy_with_logits(d_slice_old, torch.zeros_like(d_slice_old))
                            opt_d_slice.zero_grad()
                            loss_d_slice.backward()
                            opt_d_slice.step()

                            _, feat_old_for_adv = model(x_fake)
                            logits_fake2, _ = model(x_fake)
                            logits_fake2 = logits_fake2[:, :valid_out_dim]
                            slice_max_old_adv = slice_max_vector(logits_fake2, num_tasks_old, DEVICE)

                            loss_adv_feat = F.binary_cross_entropy_with_logits(feat_disc(feat_old_for_adv), torch.ones_like(feat_disc(feat_old_for_adv)))
                            loss_adv_slice = F.binary_cross_entropy_with_logits(slice_disc(slice_max_old_adv), torch.ones_like(slice_disc(slice_max_old_adv)))
                            loss = loss + lambda_adv_feat * loss_adv_feat + lambda_adv_slice * loss_adv_slice
                            loss = loss + 0.5 * criterion(logits_fake2, y_fake)

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

            protos_by_task.append(compute_prototypes(model, task_loaders, task, DEVICE))
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

        metrics = {"final_avg_class_il": float(final_avg_class), "bwt": float(bwt), "forgetting": {int(k): float(v) for k, v in forget.items()},
            "task_il_per_task": {t: {int(k): float(v) for k, v in task_il_matrix[t].items()} for t in range(5)},
            "class_il_per_task": {t: {int(k): float(v) for k, v in class_il_matrix[t].items()} for t in range(5)},}
        mgr.finish(metrics, model=model, save_model_checkpoint=save_model_checkpoint)
    return log_path


if __name__ == "__main__":
    run_experiment(run_name="exp41_dual_discriminator_ortho", config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__), vae_epochs=5, vae_n_fake_per_task=32,
        lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_ort=0.1)
