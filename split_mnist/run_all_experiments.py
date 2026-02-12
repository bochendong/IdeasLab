#!/usr/bin/env python3
# ===== 批量运行所有无 Replay 实验 =====
"""运行 exp2~exp19（Baseline 为 SplitMinist.py）。已跑过的实验会自动跳过。"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_manager import get_output_dir

# (脚本路径, 该脚本使用的 run_name，用于判断是否已跑过)
EXPERIMENTS = [
    ("split_mnist/exp2_stronger_reg.py", "exp2_stronger_reg"),
    ("split_mnist/exp3_frozen_backbone.py", "exp3_frozen_backbone"),
    ("split_mnist/exp4_ewc.py", "exp4_ewc"),
    ("split_mnist/exp5_frozen_stronger_reg.py", "exp5_frozen_stronger_reg"),
    ("split_mnist/exp6_attention_backbone.py", "exp6_attention_backbone"),
    ("split_mnist/exp7_si.py", "exp7_si"),
    ("split_mnist/exp8_attention_plus_si.py", "exp8_attention_plus_si"),
    ("split_mnist/exp9_vae_pseudo_replay.py", "exp9_vae_pseudo_replay"),
    ("split_mnist/exp10_adapters.py", "exp10_adapters"),
    ("split_mnist/exp11_slice_balance.py", "exp11_slice_balance"),
    ("split_mnist/exp12_slice_margin.py", "exp12_slice_margin"),
    ("split_mnist/exp13_task_inference.py", "exp13_task_inference"),
    # 顶会思路实验
    ("split_mnist/exp14_proto_aug.py", "exp14_proto_aug"),
    ("split_mnist/exp15_prl_base_reserve.py", "exp15_prl_base_reserve"),
    ("split_mnist/exp16_pass_ssl.py", "exp16_pass_ssl"),
    ("split_mnist/exp17_ldc_drift.py", "exp17_ldc_drift"),
    ("split_mnist/exp18_asymmetric_ce.py", "exp18_asymmetric_ce"),
    ("split_mnist/exp19_proto_aug_si.py", "exp19_proto_aug_si"),
]

EXPERIMENTS_ROOT = os.path.join(get_output_dir("split_mnist"), "experiments")


def already_run(run_name: str) -> bool:
    """若存在以 run_name 结尾且含 metrics.json 或 train.log 的实验目录，视为已跑过。"""
    if not os.path.isdir(EXPERIMENTS_ROOT):
        return False
    for name in os.listdir(EXPERIMENTS_ROOT):
        # 目录名格式: 2026-02-11_13-51-48_exp2_stronger_reg
        if not name.endswith("_" + run_name) and name != run_name:
            continue
        exp_dir = os.path.join(EXPERIMENTS_ROOT, name)
        if not os.path.isdir(exp_dir):
            continue
        if os.path.isfile(os.path.join(exp_dir, "metrics.json")) or os.path.isfile(
            os.path.join(exp_dir, "train.log")
        ):
            return True
    return False


def main():
    for exp_path, run_name in EXPERIMENTS:
        path = os.path.join(_PROJECT_ROOT, exp_path)
        if already_run(run_name):
            print(f"\n[Skip] already run: {run_name} -> {exp_path}\n")
            continue
        print(f"\n{'='*60}")
        print(f"Running: {exp_path}")
        print("="*60)
        subprocess.run([sys.executable, path], cwd=_PROJECT_ROOT)
        print(f"\nCompleted: {exp_path}\n")


if __name__ == "__main__":
    main()
