# Split CIFAR-10 任务路由 + 当前任务 margin + SI（无 replay）
# 组合：task_head(slice_vec)->task + 当前任务数据上显式 margin + SI，目标将 Task-IL 转化为 Class-IL 并推高。
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
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_d)
            break

import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment_manager import ExperimentManager

from split_cifar10.common import (
    TASK_DIR, build_all_task_loaders, cal_acc_class_il, cal_acc_class_il_routed,
    evaluate_after_task, compute_forgetting, compute_bwt,
)
from split_cifar10.base_experiment import (
    DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, CosineCNN, freeze_model,
    reg_loss_slice, log_group_differences, si_penalty, update_si_omega,
)
from split_cifar10.disc_utils import slice_max_vector, TaskFromSliceHead, NUM_TASKS


def margin_loss_current_batch(logits, y, task_id, num_tasks_cur, margin=0.3, device=DEVICE):
    slice_max_list = [logits[:, 2 * k : 2 * k + 2].max(dim=1).values for k in range(num_tasks_cur)]
    correct_slice_max = slice_max_list[task_id]
    others = [slice_max_list[k] for k in range(num_tasks_cur) if k != task_id]
    if not others:
        return torch.tensor(0.0, device=device)
    best_other = torch.stack(others, dim=1).max(dim=1).values
    gap = correct_slice_max - best_other
    return F.relu(margin - gap).mean()


def run_experiment(run_name="cifar10_task_routing_margin_si", config=None, save_model_checkpoint=False, script_file=None,
    si_lambda=0.5, lambda_task=0.5, lambda_margin=0.5, slice_margin=0.3):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    task_loaders = build_all_task_loaders(DATA_DIR, cfg["batch_size"])
    lambda_slice, lambda_feat = cfg["lambda_slice"], cfg["lambda_feat"]
    exp_config = {
        "device": str(DEVICE), "lr": cfg["lr"], "epochs_per_task": cfg["epochs_per_task"],
        "batch_size": cfg["batch_size"], "lambda_slice": lambda_slice, "lambda_feat": lambda_feat,
        "si_lambda": si_lambda, "lambda_task": lambda_task, "lambda_margin": lambda_margin,
        "slice_margin": slice_margin, "seed": cfg["seed"],
    }
    script_file = script_file or os.path.abspath(__file__)
    with ExperimentManager(run_name, exp_config, script_file=script_file, experiment_group="split_cifar10") as mgr:
        log_path = os.path.join(mgr.exp_dir, "train.log")
        mgr.set_log_path(log_path)
        setup_logging(log_path)
        logging.info("Task routing + current-task margin + SI, no replay.")
        logging.info(f"Config: {exp_config}")

        model = CosineCNN().to(DEVICE)
        task_head = TaskFromSliceHead(num_tasks=NUM_TASKS).to(DEVICE)
        prev_models = []
        si_omega_star_list = []
        task_il_matrix, class_il_matrix, class_il_routed_matrix = [], [], []

        for task in range(5):
            logging.info(f"========== Training Task {task}: classes {TASK_DIR[task]} ==========")
            optimizer = torch.optim.Adam(list(model.parameters()) + list(task_head.parameters()), lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()
            valid_out_dim = 2 * (task + 1)
            num_tasks_cur = task + 1
            si_omega = {}
            si_theta_prev = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    si_omega[n] = torch.zeros_like(p.data, device=DEVICE)
                    si_theta_prev[n] = p.data.clone()

            step = 0
            for ep in range(cfg["epochs_per_task"]):
                for x, y in task_loaders[task][0]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.train()
                    task_head.train()
                    logits, feat = model(x)
                    c_loss = criterion(logits[:, :valid_out_dim], y)
                    r_loss = reg_loss_slice(logits, feat, x, prev_models, task, lambda_slice, lambda_feat, device=DEVICE)
                    slice_vec = slice_max_vector(logits, num_tasks_cur, DEVICE)
                    task_id_labels = torch.full((x.size(0),), task, dtype=torch.long, device=DEVICE)
                    task_loss = criterion(task_head(slice_vec)[:, :num_tasks_cur], task_id_labels)
                    m_loss = lambda_margin * margin_loss_current_batch(logits, y, task, num_tasks_cur, margin=slice_margin, device=DEVICE)
                    si_loss = si_lambda * si_penalty(model, si_omega_star_list, device=DEVICE) if si_omega_star_list else torch.tensor(0.0, device=DEVICE)
                    loss = c_loss + r_loss + lambda_task * task_loss + m_loss + si_loss
                    optimizer.zero_grad()
                    loss.backward()
                    update_si_omega(si_omega, si_theta_prev, model, device=DEVICE)
                    optimizer.step()

                    if step % 50 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] loss={loss.item():.4f}"]
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(model, task_loaders[t][0], valid_out_dim, device=DEVICE)
                            acc_route = cal_acc_class_il_routed(model, task_loaders[t][1], task_head, task + 1, device=DEVICE)
                            msg.append(f"CI_T{t}={acc_ci*100:.1f}% R_T{t}={acc_route*100:.1f}%")
                        logging.info(" | ".join(msg))
                    step += 1

            si_omega_star_list.append(({n: t.clone() for n, t in si_omega.items()}, {n: p.data.clone() for n, p in model.named_parameters() if n in si_omega}))
            prev_models.append(freeze_model(copy.deepcopy(model).to(DEVICE)))

            task_il, class_il, avg_task, avg_class = evaluate_after_task(model, task, task_loaders, device=DEVICE)
            class_il_routed = {t: cal_acc_class_il_routed(model, task_loaders[t][1], task_head, task + 1, device=DEVICE) for t in range(task + 1)}
            avg_routed = float(np.mean(list(class_il_routed.values())))
            task_il_matrix.append(task_il)
            class_il_matrix.append(class_il)
            class_il_routed_matrix.append(class_il_routed)
            msg_ti = " | ".join([f"T{k}={task_il[k]*100:.2f}%" for k in range(task + 1)])
            msg_ci = " | ".join([f"T{k}={class_il[k]*100:.2f}%" for k in range(task + 1)])
            msg_route = " | ".join([f"T{k}={class_il_routed[k]*100:.2f}%" for k in range(task + 1)])
            logging.info(f"[After Task {task}] Task-IL   : {msg_ti} | Avg={avg_task*100:.2f}%")
            logging.info(f"[After Task {task}] Class-IL  : {msg_ci} | Avg={avg_class*100:.2f}%")
            logging.info(f"[After Task {task}] Class-IL(Routed): {msg_route} | Avg={avg_routed*100:.2f}%")
            log_group_differences(model, task, task_loaders)

        forget = compute_forgetting(class_il_matrix)
        bwt = compute_bwt(class_il_matrix)
        final_avg_class = float(np.mean(list(class_il_matrix[-1].values())))
        final_avg_routed = float(np.mean(list(class_il_routed_matrix[-1].values())))
        logging.info("========== FINAL METRICS ==========")
        logging.info("Forgetting per task: " + ", ".join([f"T{k}={forget[k]*100:.2f}%" for k in sorted(forget)]))
        logging.info(f"BWT: {bwt*100:.2f}%")
        logging.info(f"Final Class-IL (standard): {final_avg_class*100:.2f}%")
        logging.info(f"Final Class-IL (routed):   {final_avg_routed*100:.2f}%")
        metrics = {
            "final_avg_class_il": float(final_avg_class),
            "final_avg_class_il_routed": float(final_avg_routed),
            "bwt": float(bwt),
            "forgetting": {int(k): float(v) for k, v in forget.items()},
            "task_il_per_task": {t: {int(k): float(v) for k, v in task_il_matrix[t].items()} for t in range(5)},
            "class_il_per_task": {t: {int(k): float(v) for k, v in class_il_matrix[t].items()} for t in range(5)},
            "class_il_routed_per_task": {t: {int(k): float(v) for k, v in class_il_routed_matrix[t].items()} for t in range(5)},
        }
        mgr.finish(metrics, model=model, save_model_checkpoint=save_model_checkpoint)
    print("Done.", log_path)
    return log_path


if __name__ == "__main__":
    run_experiment(run_name="cifar10_task_routing_margin_si", script_file=os.path.abspath(__file__))
