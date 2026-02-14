# Split CIFAR-10 双判别器（对应 MNIST Exp32）
import os
import sys
import copy
import logging
import numpy as np
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if sys.platform == "win32":
    for _d in [os.path.join(_PROJECT_ROOT, ".venv", "Lib", "site-packages", "torch", "lib"), os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")]:
        if os.path.isdir(_d):
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"): os.add_dll_directory(_d)
            break

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment_manager import ExperimentManager

from split_cifar10.common import TASK_DIR, build_all_task_loaders, cal_acc_class_il, evaluate_after_task, compute_forgetting, compute_bwt
from split_cifar10.base_experiment import DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, CosineCNN, freeze_model, reg_loss_slice, log_group_differences, FEAT_DIM
from split_cifar10.vae_cifar import CVAE, train_cvae, sample_from_vaes
from split_cifar10.disc_utils import FeatDiscriminator, SliceDiscriminator, slice_max_vector, NUM_TASKS


def run_experiment(run_name="cifar10_dual_discriminator", config=None, save_model_checkpoint=False, script_file=None,
    vae_epochs=8, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    task_loaders = build_all_task_loaders(DATA_DIR, cfg["batch_size"])
    lambda_slice, lambda_feat = cfg["lambda_slice"], cfg["lambda_feat"]
    exp_config = {"device": str(DEVICE), "lr": cfg["lr"], "epochs_per_task": cfg["epochs_per_task"], "batch_size": cfg["batch_size"],
        "lambda_slice": lambda_slice, "lambda_feat": lambda_feat, "vae_epochs": vae_epochs, "vae_n_fake_per_task": vae_n_fake_per_task,
        "lambda_adv_feat": lambda_adv_feat, "lambda_adv_slice": lambda_adv_slice}
    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_cifar10") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)
        logging.info(f"Config: {exp_config}")

        model = CosineCNN().to(DEVICE)
        feat_disc = FeatDiscriminator(feat_dim=FEAT_DIM).to(DEVICE)
        slice_disc = SliceDiscriminator(input_dim=NUM_TASKS).to(DEVICE)
        prev_models, vaes = [], []
        task_il_matrix, class_il_matrix = [], []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")
            vae = CVAE(in_dim=3*32*32, num_classes=2, latent_dim=64, hidden=512).to(DEVICE)
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

                    feat_new = feat.detach()
                    slice_max_new = slice_max_vector(logits, num_tasks_cur, DEVICE)

                    if task > 0 and vaes:
                        n_fake = min(vae_n_fake_per_task * task, 128)
                        x_fake, y_fake = sample_from_vaes(vaes[:-1], list(range(task)), max(1, n_fake // task), DEVICE)
                        x_fake = x_fake.view(-1, 3, 32, 32)
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

                            _, feat_old_adv = model(x_fake)
                            logits_fake2, _ = model(x_fake)
                            logits_fake2 = logits_fake2[:, :valid_out_dim]
                            slice_max_old_adv = slice_max_vector(logits_fake2, num_tasks_old, DEVICE)
                            loss_adv_feat = F.binary_cross_entropy_with_logits(feat_disc(feat_old_adv), torch.ones_like(feat_disc(feat_old_adv)))
                            loss_adv_slice = F.binary_cross_entropy_with_logits(slice_disc(slice_max_old_adv), torch.ones_like(slice_disc(slice_max_old_adv)))
                            loss = loss + lambda_adv_feat * loss_adv_feat + lambda_adv_slice * loss_adv_slice + 0.5 * criterion(logits_fake2, y_fake)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 50 == 0:
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
        metrics = {"final_avg_class_il": float(final_avg_class), "bwt": float(bwt), "forgetting": {int(k): float(v) for k, v in forget.items()},
            "task_il_per_task": {t: {int(k): float(v) for k, v in task_il_matrix[t].items()} for t in range(5)},
            "class_il_per_task": {t: {int(k): float(v) for k, v in class_il_matrix[t].items()} for t in range(5)}}
        mgr.finish(metrics, model=model, save_model_checkpoint=save_model_checkpoint)
    print("Done.", log_path)
    return log_path

if __name__ == "__main__":
    run_experiment(run_name="cifar10_dual_discriminator", script_file=os.path.abspath(__file__))
