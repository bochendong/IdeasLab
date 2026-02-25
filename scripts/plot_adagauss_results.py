#!/usr/bin/env python3
"""
绘制 AdaGauss 实验结果曲线。
用法（在项目根目录）:
  python scripts/plot_adagauss_results.py
  python scripts/plot_adagauss_results.py --results-dir output/adagauss_results/cifar100_icarl_ada_gauss_10x10
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# 默认结果目录（取实验子目录下最新的 timestamp）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_BASE = os.path.join(PROJECT_ROOT, "output", "adagauss_results")


def find_latest_results(results_base: str):
    """在 results_base 下找到最新一次实验的 results 目录和 timestamp。"""
    # 例如 output/adagauss_results/cifar100_icarl_ada_gauss_10x10
    exp_dirs = [
        d for d in glob.glob(os.path.join(results_base, "*"))
        if os.path.isdir(d)
    ]
    if not exp_dirs:
        return None, None
    # 取第一个实验目录（通常只有一个）
    exp_dir = exp_dirs[0]
    results_dir = os.path.join(exp_dir, "results")
    if not os.path.isdir(results_dir):
        return None, None
    # 用 acc_tag 找最新 timestamp
    pattern = os.path.join(results_dir, "acc_tag-*.txt")
    files = glob.glob(pattern)
    if not files:
        return None, None
    # 从文件名取 timestamp，选最新的
    parts = [os.path.basename(f).replace("acc_tag-", "").replace(".txt", "") for f in files]
    latest = max(parts)
    return exp_dir, latest


def load_matrix(path: str):
    """加载制表符分隔的矩阵，返回 numpy 数组。"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows) if rows else None


def load_vector(path: str):
    """加载单行制表符分隔的向量。"""
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return np.array([float(x) for x in line.split()])


def plot_adagauss_curves(exp_dir: str, timestamp: str, out_path: str = None, show: bool = True):
    results_dir = os.path.join(exp_dir, "results")
    prefix = os.path.join(results_dir, f"{{}}-{timestamp}.txt")

    avg_taw = load_vector(prefix.format("avg_accs_taw"))
    avg_tag = load_vector(prefix.format("avg_accs_tag"))
    acc_taw = load_matrix(prefix.format("acc_taw"))
    acc_tag = load_matrix(prefix.format("acc_tag"))

    if avg_taw is None or avg_tag is None:
        raise FileNotFoundError(f"找不到 avg 文件，请检查 {results_dir} 与 timestamp {timestamp}")

    num_tasks = len(avg_taw)
    tasks = np.arange(1, num_tasks + 1, dtype=int)

    # 1) 增量学习曲线：平均准确率 vs 已学任务数
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(tasks, 100 * avg_taw, "o-", color="C0", label="TAw (Task-Aware)", linewidth=2, markersize=8)
    ax.plot(tasks, 100 * avg_tag, "s-", color="C1", label="TAg (Task-Agnostic, Class-IL)", linewidth=2, markersize=8)
    ax.set_xlabel("Number of tasks learned", fontsize=11)
    ax.set_ylabel("Average accuracy (%)", fontsize=11)
    ax.set_title("Incremental learning: avg accuracy vs tasks", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(tasks)
    ax.set_ylim(0, 105)

    # 2) 学完 10 个任务后，各任务上的 TAg 准确率（柱状图）
    ax = axes[1]
    if acc_tag is not None and acc_tag.shape[0] >= num_tasks:
        final_row = acc_tag[num_tasks - 1, :num_tasks]
        ax.bar(tasks, 100 * final_row, color="steelblue", edgecolor="navy", alpha=0.85)
        ax.axhline(100 * avg_tag[-1], color="red", linestyle="--", linewidth=1.5, label=f"Avg TAg = {100*avg_tag[-1]:.2f}%")
        ax.set_xlabel("Task ID", fontsize=11)
        ax.set_ylabel("TAg accuracy (%)", fontsize=11)
        ax.set_title("Per-task Class-IL accuracy (after 10 tasks)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(tasks)
        ax.set_ylim(0, 105)
    else:
        ax.text(0.5, 0.5, "No acc_tag matrix", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-task TAg accuracy", fontsize=12)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved:", out_path)
    else:
        out_path = os.path.join(exp_dir, "adagauss_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved:", out_path)
    if show:
        plt.show()
    else:
        plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="绘制 AdaGauss 结果曲线")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="实验目录，例如 output/adagauss_results/cifar100_icarl_ada_gauss_10x10；不填则自动找最新一次",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出图片路径；不填则保存在实验目录下 adagauss_curves.png",
    )
    parser.add_argument("--no-show", action="store_true", help="不弹出显示窗口，只保存文件")
    args = parser.parse_args()

    if args.no_show:
        plt.switch_backend("Agg")

    if args.results_dir:
        exp_dir = os.path.abspath(args.results_dir)
        results_dir = os.path.join(exp_dir, "results")
        if not os.path.isdir(results_dir):
            raise SystemExit(f"目录不存在或没有 results 子目录: {exp_dir}")
        acc_tag_files = glob.glob(os.path.join(results_dir, "acc_tag-*.txt"))
        if not acc_tag_files:
            raise SystemExit(f"未找到 acc_tag-*.txt: {results_dir}")
        timestamp = os.path.basename(acc_tag_files[0]).replace("acc_tag-", "").replace(".txt", "")
        # 若有多份，用最新的
        for f in acc_tag_files:
            ts = os.path.basename(f).replace("acc_tag-", "").replace(".txt", "")
            if ts > timestamp:
                timestamp = ts
    else:
        exp_dir, timestamp = find_latest_results(DEFAULT_RESULTS_BASE)
        if exp_dir is None:
            raise SystemExit(f"未找到任何实验结果，请先运行 AdaGauss 或指定 --results-dir。目录: {DEFAULT_RESULTS_BASE}")

    plot_adagauss_curves(exp_dir, timestamp, out_path=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
