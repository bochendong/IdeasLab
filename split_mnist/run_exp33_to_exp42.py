#!/usr/bin/env python3
# ===== 只跑 Exp33～Exp42（冲 50%+ 与 doc 借鉴实验）=====
"""依次运行 Exp33、34、35、36、37、38、39、40、41、42。"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_manager import get_output_dir

EXPERIMENTS = [
    ("split_mnist/exp33_dual_discriminator_si.py", "exp33_dual_discriminator_si"),
    ("split_mnist/exp34_dual_discriminator_stronger_replay.py", "exp34_dual_discriminator_stronger_replay"),
    ("split_mnist/exp35_dual_discriminator_slice_margin.py", "exp35_dual_discriminator_slice_margin"),
    ("split_mnist/exp36_dual_discriminator_proto.py", "exp36_dual_discriminator_proto"),
    ("split_mnist/exp37_dual_discriminator_balanced_batch.py", "exp37_dual_discriminator_balanced_batch"),
    ("split_mnist/exp38_dual_discriminator_slice_var.py", "exp38_dual_discriminator_slice_var"),
    ("split_mnist/exp39_dual_discriminator_anticollapse.py", "exp39_dual_discriminator_anticollapse"),
    ("split_mnist/exp40_dual_discriminator_efm.py", "exp40_dual_discriminator_efm"),
    ("split_mnist/exp41_dual_discriminator_ortho.py", "exp41_dual_discriminator_ortho"),
    ("split_mnist/exp42_dual_discriminator_slice_consist.py", "exp42_dual_discriminator_slice_consist"),
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
