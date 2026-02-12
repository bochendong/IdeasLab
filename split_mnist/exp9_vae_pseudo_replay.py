# ===== Exp9: VAE 生成式伪样本（无真实回放存储）=====
"""每任务训练一个小型 CVAE，新任务训练时从旧任务 VAE 采样作为伪样本参与 CE 损失，不存储任何真实样本。"""
import os
import sys
import copy
import random
import logging
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows: PyTorch DLL 搜索路径，避免 c10.dll 加载失败
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
from torch.distributions import Normal
from experiment_manager import ExperimentManager, get_output_dir

from split_mnist.common import (
    TASK_DIR,
    build_all_task_loaders,
    cal_acc_class_il,
    evaluate_after_task,
    compute_forgetting,
    compute_bwt,
)
from split_mnist.base_experiment import (
    DEFAULT_CONFIG,
    DEVICE,
    DATA_DIR,
    set_seed,
    setup_logging,
    CosineMLP,
    freeze_model,
    get_backbone_params,
    reg_loss_slice,
    log_group_differences,
)

OUTPUT_DIR = get_output_dir("split_mnist")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------
# CVAE：条件 VAE，每任务 2 类
# ------------------------
class CVAE(nn.Module):
    def __init__(self, in_dim=784, num_classes=2, latent_dim=32, hidden=256):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(num_classes, 16)
        self.enc = nn.Sequential(
            nn.Linear(in_dim + 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
            nn.Tanh(),
        )

    def encode(self, x, y):
        # y: (B,) in [0, 1]
        emb = self.class_emb(y)
        h = torch.cat([x.view(-1, self.in_dim), emb], dim=1)
        h = self.enc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        emb = self.class_emb(y)
        h = torch.cat([z, emb], dim=1)
        return self.dec(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=device, dtype=torch.long)
        x = self.decode(z, y)
        return x, y


def train_cvae(vae, loader, device, epochs=5, lr=1e-3):
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device).view(-1, 784)
            # 全局类 0,1 -> 局部 0,1（task 0）；其他 task 同理，这里每个 loader 已是单任务，y 已是 0,1 或 2,3...
            y_local = (y % 2).long().to(device)  # 0,1 -> 0,1; 2,3 -> 0,1
            recon, mu, logvar = vae(x, y_local)
            recon_loss = F.mse_loss(recon, x)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            loss = recon_loss + 0.1 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return vae


def sample_from_vaes(vaes, task_ids, n_per_task, device, task_dir=None):
    """从多个任务的 CVAE 中各采样 n_per_task 个 (x, y_global)。"""
    task_dir = task_dir or TASK_DIR
    xs, ys = [], []
    for vae, tid in zip(vaes, task_ids):
        x, y_local = vae.sample(n_per_task, device)
        # 局部类 0/1 -> 全局类
        y_global = torch.tensor(
            [task_dir[int(tid)][int(b)] for b in y_local.cpu().numpy()],
            device=device,
            dtype=torch.long,
        )
        xs.append(x)
        ys.append(y_global)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def run_experiment_vae(
    run_name: str = "exp9_vae_pseudo_replay",
    config: dict = None,
    save_model_checkpoint: bool = False,
    script_file: str = None,
    vae_epochs: int = 5,
    vae_n_fake_per_task: int = 32,
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
        "use_cosine_head": cfg["use_cosine_head"],
        "seed": cfg["seed"],
        "vae_epochs": vae_epochs,
        "vae_n_fake_per_task": vae_n_fake_per_task,
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
        vaes = []  # 每任务一个 CVAE，不存真实数据

        task_il_matrix = []
        class_il_matrix = []

        for task in range(5):
            logging.info(f"========== Task {task}: classes {TASK_DIR[task]} ==========")

            # 1) 在当前任务数据上训练 CVAE
            vae = CVAE(in_dim=784, num_classes=2, latent_dim=32, hidden=256).to(DEVICE)
            vae = train_cvae(vae, task_loaders[task][0], DEVICE, epochs=vae_epochs)
            vaes.append(copy.deepcopy(vae))
            vae = vaes[-1]

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
                    loss = c_loss + r_loss

                    # 从旧任务 VAE 采样伪样本（无真实回放）
                    if task > 0 and vaes:
                        n_fake = min(vae_n_fake_per_task * task, 128)
                        x_fake, y_fake = sample_from_vaes(vaes[:-1], list(range(task)), n_fake // task, DEVICE)
                        x_fake = x_fake.view(-1, 1, 28, 28)
                        logits_fake, _ = model(x_fake)
                        loss_fake = criterion(logits_fake[:, :valid_out_dim], y_fake)
                        loss = loss + 0.5 * loss_fake

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
    run_experiment_vae(
        run_name="exp9_vae_pseudo_replay",
        config={"lambda_slice": 2.0, "lambda_feat": 0.5},
        script_file=os.path.abspath(__file__),
        vae_epochs=5,
        vae_n_fake_per_task=32,
    )
