#!/usr/bin/env python3
# ===== 批量运行所有无 Replay 实验 =====
"""运行 exp1~exp4（Baseline 为 SplitMinist.py）"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

EXPERIMENTS = [
    "split_mnist/exp2_stronger_reg.py",
    "split_mnist/exp3_frozen_backbone.py",
    "split_mnist/exp4_ewc.py",
    "split_mnist/exp5_frozen_stronger_reg.py",
]


def main():
    for exp in EXPERIMENTS:
        path = os.path.join(_PROJECT_ROOT, exp)
        print(f"\n{'='*60}")
        print(f"Running: {exp}")
        print("="*60)
        subprocess.run([sys.executable, path], cwd=_PROJECT_ROOT)
        print(f"\nCompleted: {exp}\n")


if __name__ == "__main__":
    main()
