# ===== CIFAR-10 Baseline 超参扫描：先调出较好 baseline，再固定 lr/epochs 跑其他方法 =====
# 用法（项目根目录）：python split_cifar10/tune_baseline.py
# 可选：python split_cifar10/tune_baseline.py --lr 0.001 0.003 --ep 10 12  （只扫指定 lr/ep）
import os
import sys
import json
import argparse
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from split_cifar10.exp_baseline import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Tune CIFAR-10 baseline (lr, epochs_per_task). Same config then used for all other methods.")
    parser.add_argument("--lr", nargs="*", type=float, default=[1e-3, 3e-3, 5e-3], help="Learning rates to try.")
    parser.add_argument("--ep", nargs="*", type=int, default=[8, 10, 12], help="Epochs per task to try.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (fixed).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    tune_id = datetime.now().strftime("%Y%m%d_%H%M")
    lrs = args.lr
    eps = args.ep
    configs = []
    for lr in lrs:
        for ep in eps:
            configs.append({"lr": lr, "epochs_per_task": ep, "batch_size": args.batch_size, "seed": args.seed})

    print(f"Tune ID: {tune_id} | Configs: {len(configs)} (lr={lrs}, ep={eps})")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for c in configs:
        lr, ep = c["lr"], c["epochs_per_task"]
        run_name = f"cifar10_baseline_tune_{tune_id}_lr{lr}_ep{ep}"
        print(f"\n>> Running {run_name}")
        run_experiment(run_name=run_name, config=c, save_model_checkpoint=False, script_file=os.path.join(script_dir, "exp_baseline.py"))

    # 汇总：找到本轮所有 baseline_tune 结果并打表
    exp_root = os.path.join(_PROJECT_ROOT, "output", "split_cifar10", "experiments")
    rows = []
    for d in os.listdir(exp_root):
        if tune_id not in d or "baseline_tune" not in d:
            continue
        path = os.path.join(exp_root, d)
        if not os.path.isdir(path):
            continue
        cfg_path = os.path.join(path, "config.json")
        met_path = os.path.join(path, "metrics.json")
        if not os.path.isfile(met_path):
            continue
        with open(met_path, "r", encoding="utf-8") as f:
            met = json.load(f)
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        lr = cfg.get("lr", "")
        ep = cfg.get("epochs_per_task", "")
        rows.append({
            "lr": lr,
            "ep": ep,
            "class_il": met.get("final_avg_class_il", 0) * 100,
            "bwt": met.get("bwt", 0) * 100,
            "dir": d,
        })
    rows.sort(key=lambda x: (-x["class_il"], x["bwt"]))  # Class-IL 高优先，同分看 BWT 略好

    print("\n" + "=" * 60)
    print("BASELINE TUNE SUMMARY (higher Class-IL better)")
    print("=" * 60)
    if not rows:
        print("No metrics found. Check output/split_cifar10/experiments/")
        return
    for r in rows:
        print(f"  lr={r['lr']}  ep={r['ep']}  ->  Class-IL={r['class_il']:.2f}%  BWT={r['bwt']:.2f}%  |  {r['dir']}")
    best = rows[0]
    print("=" * 60)
    print(f"Suggested fixed config for all methods:  lr={best['lr']}, epochs_per_task={best['ep']}, batch_size={args.batch_size}, seed={args.seed}")
    print("Update split_cifar10/base_experiment.py DEFAULT_CONFIG or pass config= in each experiment.")


if __name__ == "__main__":
    main()
