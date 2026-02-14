#!/usr/bin/env python3
# ===== 只跑 Exp45～Exp49（机器学习方法结合实验）=====
"""Exp45 Focal Loss | Exp46 软标签蒸馏 | Exp47 Mixup | Exp48 对比学习（原型）| Exp49 重要性加权。基线均为 Exp35（双判别器+slice margin）。"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_manager import get_output_dir

EXPERIMENTS = [
    ("split_mnist/exp45_slice_margin_focal.py", "exp45_slice_margin_focal"),
    ("split_mnist/exp46_slice_margin_distill.py", "exp46_slice_margin_distill"),
    ("split_mnist/exp47_slice_margin_mixup.py", "exp47_slice_margin_mixup"),
    ("split_mnist/exp48_slice_margin_contrastive.py", "exp48_slice_margin_contrastive"),
    ("split_mnist/exp49_slice_margin_importance.py", "exp49_slice_margin_importance"),
]

EXPERIMENTS_ROOT = os.path.join(get_output_dir("split_mnist"), "experiments")


def already_run(run_name: str) -> bool:
    if not os.path.isdir(EXPERIMENTS_ROOT):
        return False
    for name in os.listdir(EXPERIMENTS_ROOT):
        if not name.endswith("_" + run_name) and name != run_name:
            continue
        exp_dir = os.path.join(EXPERIMENTS_ROOT, name)
        if not os.path.isdir(exp_dir):
            continue
        if os.path.isfile(os.path.join(exp_dir, "metrics.json")) or os.path.isfile(os.path.join(exp_dir, "train.log")):
            return True
    return False


def main():
    for exp_path, run_name in EXPERIMENTS:
        path = os.path.join(_PROJECT_ROOT, exp_path)
        if not os.path.isfile(path):
            print(f"[Warn] Not found: {path}")
            continue
        if already_run(run_name):
            print(f"[Skip] already run: {run_name}")
            continue
        print(f"\n{'='*60}\nRunning: {run_name}\n{'='*60}")
        ret = subprocess.run([sys.executable, path], cwd=_PROJECT_ROOT)
        print(f"\nDone: {run_name} (exit {ret.returncode})\n")
    print("All done.")


if __name__ == "__main__":
    main()
