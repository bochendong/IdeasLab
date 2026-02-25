#!/usr/bin/env python3
# 在项目根目录执行: python split_cifar100/run_baseline.py
# 快速试跑（少 epoch）: python split_cifar100/run_baseline.py --epochs 5
"""Split CIFAR-100 10×10 基线：多任务 head + CE，评估输出 TAw/TAg（与 AdaGauss 一致）。"""
import argparse
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from split_cifar100.base_experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="epochs per task")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "cnn"],
                        help="resnet18=与 AdaGauss 一致 (默认); cnn=轻量 CNN 快速试跑")
    args = parser.parse_args()

    run_experiment(
        run_name="cifar100_baseline",
        config={
            "epochs_per_task": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "backbone": args.backbone,
        },
        save_model_checkpoint=False,
        script_file=__file__,
    )
