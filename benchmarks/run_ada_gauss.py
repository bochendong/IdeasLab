#!/usr/bin/env python3
# ===== 与 MNIST 一致：用当前 Python 解释器跑 AdaGauss =====
# 在项目根目录执行: python benchmarks/run_ada_gauss.py
# 完整复现: python benchmarks/run_ada_gauss.py --full
"""
用 sys.executable 调用 AdaGauss，和 split_mnist/run_all_experiments.py 一样，
保证用的是你当前激活的 Python（conda/venv 或 PATH 里的 python）。
"""
import argparse
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ADAGAUSS = os.path.join(_PROJECT_ROOT, "benchmarks", "AdaGauss")
_SRC = os.path.join(_ADAGAUSS, "src")
_MAIN = os.path.join(_SRC, "main_incremental.py")


def main():
    parser = argparse.ArgumentParser(description="Run AdaGauss (same Python as MNIST)")
    parser.add_argument("--full", action="store_true", help="200 epochs/task (default: 2 for quick)")
    args = parser.parse_args()

    if not os.path.isfile(_MAIN):
        print(f"AdaGauss not found at {_ADAGAUSS}. Run: git clone https://github.com/grypesc/AdaGauss.git benchmarks/AdaGauss")
        sys.exit(1)

    out_root = os.path.join(_PROJECT_ROOT, "output")
    os.makedirs(out_root, exist_ok=True)

    if args.full:
        results_path = os.path.join(out_root, "adagauss_results")
        print("Running AdaGauss full (200 epochs/task)...")
        cmd = [
            sys.executable, _MAIN,
            "--approach", "ada_gauss", "--seed", "1", "--batch-size", "256",
            "--num-workers", "4", "--nepochs", "200", "--datasets", "cifar100_icarl",
            "--num-tasks", "10", "--nc-first-task", "10", "--lr", "0.1",
            "--weight-decay", "5e-4", "--adaptation-strategy", "full", "--S", "64",
            "--lamb", "10", "--use-test-as-val", "--criterion", "ce",
            "--distillation", "projected", "--rotation", "--normalize",
            "--multiplier", "32", "--distiller", "mlp", "--adapter", "mlp",
            "--exp-name", "10x10/", "--results-path", results_path,
        ]
    else:
        results_path = os.path.join(out_root, "adagauss_quick")
        print("Running AdaGauss quick (2 epochs/task)...")
        cmd = [
            sys.executable, _MAIN,
            "--approach", "ada_gauss", "--seed", "1", "--batch-size", "128",
            "--num-workers", "2", "--nepochs", "2", "--datasets", "cifar100_icarl",
            "--num-tasks", "10", "--nc-first-task", "10", "--lr", "0.1",
            "--weight-decay", "5e-4", "--adaptation-strategy", "full", "--S", "64",
            "--lamb", "10", "--use-test-as-val", "--criterion", "ce",
            "--distillation", "projected", "--rotation", "--normalize",
            "--multiplier", "32", "--distiller", "mlp", "--adapter", "mlp",
            "--exp-name", "quick/", "--results-path", results_path,
        ]

    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC

    ret = subprocess.run(cmd, cwd=_ADAGAUSS, env=env)
    print("Results under:", results_path)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
