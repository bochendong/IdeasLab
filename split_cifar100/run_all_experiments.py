#!/usr/bin/env python3
# 在项目根目录执行: python split_cifar100/run_all_experiments.py
# 快速试跑（少 epoch）: python split_cifar100/run_all_experiments.py --epochs 5
"""
按 [docs/mnist_cifar100_combination.md] 实现并记录 CIFAR-100 实验：
baseline, slice_margin, slice_margin_feat_kd, feat_kd, logit_kd, ewc, si, slice_margin_ewc
每实验独立 run_name，结果与 config 自动写入 output/split_cifar100/experiments/。
"""
import argparse
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from split_cifar100.base_experiment import run_experiment

# 实验注册表：run_name -> config（覆盖 DEFAULT_CONFIG 的项）
EXPERIMENTS = {
    "cifar100_baseline": {
        "description": "ResNet18 + 10 head, CE only",
        "config": {},
    },
    "cifar100_slice_margin": {
        "description": "Slice margin (correct task head max > others + margin)",
        "config": {
            "lambda_slice_margin": 0.5,
            "slice_margin": 0.5,
        },
    },
    "cifar100_feat_kd": {
        "description": "Feature distillation: MSE(new_feat, old_feat)",
        "config": {
            "lambda_feat_kd": 1.0,
        },
    },
    "cifar100_logit_kd": {
        "description": "Logit distillation (LwF style) on old heads",
        "config": {
            "lambda_logit_kd": 2.0,
            "tau_kd": 2.0,
        },
    },
    "cifar100_slice_margin_feat_kd": {
        "description": "Slice margin + feature KD",
        "config": {
            "lambda_slice_margin": 0.5,
            "slice_margin": 0.5,
            "lambda_feat_kd": 1.0,
        },
    },
    "cifar100_slice_margin_logit_kd": {
        "description": "Slice margin + logit KD",
        "config": {
            "lambda_slice_margin": 0.5,
            "slice_margin": 0.5,
            "lambda_logit_kd": 2.0,
            "tau_kd": 2.0,
        },
    },
    "cifar100_ewc": {
        "description": "EWC on backbone + heads",
        "config": {
            "ewc_lambda": 1000.0,
        },
    },
    "cifar100_si": {
        "description": "Synaptic Intelligence",
        "config": {
            "si_lambda": 0.5,
        },
    },
    "cifar100_slice_margin_ewc": {
        "description": "Slice margin + EWC",
        "config": {
            "lambda_slice_margin": 0.5,
            "slice_margin": 0.5,
            "ewc_lambda": 1000.0,
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-100 experiments (split_cifar100)")
    parser.add_argument("--epochs", type=int, default=50, help="epochs per task")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="run only these (default: all). e.g. cifar100_baseline cifar100_slice_margin")
    parser.add_argument("--list", action="store_true", help="list experiment names and exit")
    args = parser.parse_args()

    if args.list:
        for name, rec in EXPERIMENTS.items():
            print(f"  {name}: {rec['description']}")
        return

    base = {"epochs_per_task": args.epochs, "lr": args.lr, "seed": args.seed}
    to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())
    for run_name in to_run:
        if run_name not in EXPERIMENTS:
            print(f"Unknown experiment: {run_name}")
            continue
        rec = EXPERIMENTS[run_name]
        config = {**base, **rec["config"]}
        print(f"\n>>> Running: {run_name} ({rec['description']})")
        run_experiment(
            run_name=run_name,
            config=config,
            save_model_checkpoint=False,
            script_file=__file__,
        )
    print("\nDone. Results under output/split_cifar100/experiments/")


if __name__ == "__main__":
    main()
