# CIFAR-10 条件 VAE（in_dim=3072，每任务 2 类）
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_cifar10.common import TASK_DIR

# CIFAR-10 展平维度
CIFAR_IN_DIM = 3 * 32 * 32  # 3072


class CVAE(nn.Module):
    def __init__(self, in_dim=3072, num_classes=2, latent_dim=64, hidden=512):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(num_classes, 32)
        self.enc = nn.Sequential(
            nn.Linear(in_dim + 32, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
            nn.Tanh(),
        )

    def encode(self, x, y):
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


def train_cvae(vae, loader, device, epochs=8, lr=1e-3):
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device).view(-1, CIFAR_IN_DIM)
            y_local = (y % 2).long().to(device)
            recon, mu, logvar = vae(x, y_local)
            recon_loss = F.mse_loss(recon, x)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            loss = recon_loss + 0.1 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return vae


def sample_from_vaes(vaes, task_ids, n_per_task, device, task_dir=None):
    task_dir = task_dir or TASK_DIR
    xs, ys = [], []
    for vae, tid in zip(vaes, task_ids):
        x, y_local = vae.sample(n_per_task, device)
        y_global = torch.tensor(
            [task_dir[int(tid)][int(b)] for b in y_local.cpu().numpy()],
            device=device, dtype=torch.long,
        )
        xs.append(x)
        ys.append(y_global)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
