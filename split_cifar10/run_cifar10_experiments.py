# ===== 在 Split CIFAR-10 上跑效果好的实验 =====
# 用法（在项目根目录）：python split_cifar10/run_cifar10_experiments.py
# 或只跑部分：python split_cifar10/run_cifar10_experiments.py --only baseline slice_margin
import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from split_cifar10.exp_baseline import run_experiment as run_baseline
from split_cifar10.exp_ewc import run_experiment as run_ewc
from split_cifar10.exp_si import run_experiment as run_si
from split_cifar10.exp_vae_si import run_experiment as run_vae_si
from split_cifar10.exp_dual_discriminator import run_experiment as run_dual_disc
from split_cifar10.exp_slice_margin import run_experiment as run_slice_margin
from split_cifar10.exp_stronger_replay import run_experiment as run_stronger_replay
from split_cifar10.exp_task_routing_si import run_experiment as run_task_routing_si
from split_cifar10.exp_current_margin_si import run_experiment as run_current_margin_si
from split_cifar10.exp_task_routing_margin_si import run_experiment as run_task_routing_margin_si


EXPERIMENTS = {
    "baseline": ("Baseline (no reg)", run_baseline, {}),
    "ewc": ("EWC", run_ewc, {}),
    "si": ("SI", run_si, {}),
    "vae_si": ("VAE pseudo-replay + SI", run_vae_si, {}),
    "dual_discriminator": ("Dual discriminator", run_dual_disc, {}),
    "slice_margin": ("Dual disc + Slice margin", run_slice_margin, {}),
    "stronger_replay": ("Stronger replay", run_stronger_replay, {}),
    # 无 replay：利用 Task-IL 保持 → 推高 Class-IL（任务路由 / 当前任务 margin）
    "task_routing_si": ("Task-from-slice routing + SI, no replay", run_task_routing_si, {}),
    "current_margin_si": ("Current-task slice margin + SI, no replay", run_current_margin_si, {}),
    "task_routing_margin_si": ("Task routing + current margin + SI, no replay", run_task_routing_margin_si, {}),
}


def main():
    parser = argparse.ArgumentParser(description="Run Split CIFAR-10 experiments (good methods from MNIST).")
    parser.add_argument("--only", nargs="*", default=None, help="Only run these experiments (e.g. baseline slice_margin). Default: run all.")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size (default 256). Use 512 if GPU has headroom, 128 or 64 if OOM.")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for key, (desc, _, _) in EXPERIMENTS.items():
            print(f"  {key}: {desc}")
        return

    to_run = list(EXPERIMENTS.keys()) if (not args.only or len(args.only) == 0) else args.only
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for key in to_run:
        if key not in EXPERIMENTS:
            print(f"Unknown experiment: {key}. Use --list to see available.")
            continue
        desc, run_fn, kwargs = EXPERIMENTS[key]
        if args.batch_size is not None:
            kwargs = {**kwargs, "config": {**(kwargs.get("config") or {}), "batch_size": args.batch_size}}
        print(f"\n{'='*60}\nRunning: {key} ({desc})\n{'='*60}")
        run_name = f"cifar10_{key}"
        script_file = os.path.join(script_dir, f"exp_{key}.py")
        run_fn(run_name=run_name, script_file=script_file, **kwargs)
    print("\nAll selected experiments finished. Results: output/split_cifar10/experiments/")


if __name__ == "__main__":
    main()
