# ===== Exp14: PASS 风格原型增强（ProtoAug）=====
"""只存每类特征均值（原型），新任务训练时对旧类原型加高斯噪声生成伪特征参与 CE，无 VAE、无真实回放。"""
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
from split_mnist.base_experiment import DEFAULT_CONFIG, DEVICE, DATA_DIR, set_seed, setup_logging, CosineMLP, freeze_model, reg_loss_slice, log_group_differences

OUTPUT_DIR = get_output_dir("split_mnist")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_prototypes(model, task_loaders, task_id, device, featurizer=None):
    """计算 task_id 各类的特征均值（原型）。featurizer 若为 None 则用 model 的 backbone 输出。"""
    model.eval()
    featurizer = featurizer or (lambda x: model(x)[1])
    protos = {}  # class_id -> (B,) feature mean
    feat_by_class = {c: [] for c in TASK_DIR[task_id]}

    with torch.no_grad():
        for x, y in task_loaders[task_id][0]:
            x = x.to(device)
            feat = featurizer(x)
            for i in range(x.size(0)):
                c = int(y[i].item())
                feat_by_class[c].append(feat[i])

    for c in TASK_DIR[task_id]:
        if feat_by_class[c]:
            protos[c] = torch.stack(feat_by_class[c]).mean(dim=0)
        else:
            protos[c] = torch.zeros(feat.size(1), device=device)
    return protos


def sample_augmented_prototypes(protos_dict, n_per_class, r_scale, device):
    """从原型采样：proto + r * randn。protos_dict: {class_id: tensor(400)}"""
    feats, labels = [], []
    for c, proto in protos_dict.items():
        for _ in range(n_per_class):
            noise = torch.randn_like(proto, device=device)
            std = max(proto.std().item(), 1e-6)
            aug = proto + r_scale * std * noise
            feats.append(aug)
            labels.append(c)
    if not feats:
        return None, None
    return torch.stack(feats), torch.tensor(labels, device=device, dtype=torch.long)


def run_experiment_proto_aug(
    run_name: str = "exp14_proto_aug",
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
    proto_aug_r: float = 0.4,
    proto_aug_n_per_class: int = 8,
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
        all_protos = {}  # task_id -> {class_id: prototype}

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

                    # 从旧任务原型采样伪特征参与 CE（all_protos 为之前任务已存的原型）
                    if task > 0 and all_protos:
                        old_protos = {c: p for c, p in all_protos.items() if c not in TASK_DIR[task]}
                        if old_protos:
                            n_old = len(old_protos)
                            n_per = max(1, proto_aug_n_per_class)
                            feat_aug, label_aug = sample_augmented_prototypes(old_protos, n_per, proto_aug_r, DEVICE)
                            if feat_aug is not None:
                                # 伪特征过 head 得 logits（CosineMLP: logit_scale * feat_n @ W_n.t）
                                feat_n = F.normalize(feat_aug, dim=1)
                                W_n = F.normalize(model.W.data[:valid_out_dim].t(), dim=1)
                                logits_aug = model.logit_scale * (feat_n @ W_n)
                                loss_aug = criterion(logits_aug, label_aug)
                                loss = loss + 0.5 * loss_aug

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 25 == 0:
                        msg = [f"[Task {task} | ep {ep} | step {step}] loss={loss.item():.4f} c={c_loss.item():.4f} r={r_loss.item():.4f}"]
                        for t in range(task + 1):
                            acc_ci = cal_acc_class_il(model, task_loaders[t][0], valid_out_dim, device=DEVICE)
                            msg.append(f"CI_T{t}={acc_ci*100:.2f}%")
                        logging.info(" | ".join(msg))
                    step += 1

            # 任务结束后计算并存储当前任务各类原型
            protos_t = compute_prototypes(model, task_loaders, task, DEVICE)
            for c, p in protos_t.items():
                all_protos[c] = p.detach().clone()

            teacher = copy.deepcopy(model).to(DEVICE)
            prev_models.append(freeze_model(teacher))

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
    run_experiment_proto_aug(
        run_name="exp14_proto_aug",
        config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__),
        proto_aug_r=0.4,
        proto_aug_n_per_class=8,
    )
