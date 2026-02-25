#!/usr/bin/env python3
# ===== 聚合 MNIST / CIFAR-100 实验结果：横向对比 + 超参 =====
"""
扫描 output/split_mnist/experiments 与 output/split_cifar100/experiments，
生成：
1) MNIST vs CIFAR-100 横向对比表（同一方法并排）
2) 全部实验表（含主要超参：lr, epochs, margin, lambda_* 等）

用法（项目根目录）：
  python scripts/aggregate_results.py                    # 输出到 docs/experiment_results.md
  python scripts/aggregate_results.py --update-readme     # 同时更新 README 中的对比表块
"""
import os
import sys
import json
import argparse
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "output")
README_PATH = os.path.join(_PROJECT_ROOT, "README.md")
DOCS_DIR = os.path.join(_PROJECT_ROOT, "docs")
RESULTS_MD = os.path.join(DOCS_DIR, "experiment_results.md")

# 用于横向对比的 run_name -> 统一方法键（便于 MNIST 与 CIFAR-100 并排）
RUN_NAME_TO_CROSS_KEY = {
    # CIFAR-100（split_cifar100 实验，run_all_experiments.py）
    "cifar100_baseline": "baseline",
    "cifar100_slice_margin": "slice_margin",
    "cifar100_feat_kd": "feat_kd",
    "cifar100_logit_kd": "logit_kd",
    "cifar100_slice_margin_feat_kd": "slice_margin_feat_kd",
    "cifar100_slice_margin_logit_kd": "slice_margin_logit_kd",
    "cifar100_ewc": "ewc",
    "cifar100_si": "si",
    "cifar100_slice_margin_ewc": "slice_margin_ewc",
    "cifar100_dual_discriminator": "dual_discriminator",
    # MNIST
    "exp35_dual_discriminator_slice_margin": "slice_margin",
    "exp34_dual_discriminator_stronger_replay": "stronger_replay",
    "exp32_dual_discriminator": "dual_discriminator",
    "exp27_vae_pseudo_replay_si": "vae_si",
    "exp4_ewc": "ewc",
    "exp7_si": "si",
}

# 方法键 -> 显示名（横向对比表用）
CROSS_KEY_TO_DISPLAY = {
    "baseline": "Baseline",
    "ewc": "EWC",
    "si": "SI",
    "vae_si": "VAE+SI",
    "dual_discriminator": "Dual discriminator",
    "slice_margin": "Slice margin",
    "slice_margin_feat_kd": "Slice margin+Feat KD",
    "slice_margin_logit_kd": "Slice margin+Logit KD",
    "slice_margin_ewc": "Slice margin+EWC",
    "feat_kd": "Feat KD",
    "logit_kd": "Logit KD",
    "stronger_replay": "Stronger replay",
}

# MNIST baseline 若没有单独 run，用 README 中记录值
MNIST_BASELINE_FALLBACK = (25.69, -92.21)


def _load_latest_per_run(experiment_group: str):
    """扫描 output/<group>/experiments/，每个 run_name 只保留最新一次（按目录名时间戳）。返回 [(run_name, metrics, config, exp_dir), ...]"""
    root = os.path.join(OUTPUT_ROOT, experiment_group, "experiments")
    if not os.path.isdir(root):
        return []
    by_run = {}
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 目录名格式: 2026-02-12_23-32-55_exp35_dual_discriminator_slice_margin 或 2026-02-13_16-23-50_cifar10_slice_margin
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        ts, run_part = parts[0], parts[1]
        # 去掉可能的时间前缀得到纯 run_name。格式为 "HH-MM-SS_xxx" 或 "H1-H2-H3_exp35_xxx"（三段数字）
        first = run_part.split("_", 1)[0]
        if "_" in run_part and len(first) == 8 and first.replace("-", "").isdigit():
            run_name = run_part.split("_", 1)[1]  # HH-MM-SS_rest -> rest
        else:
            segs = run_part.split("_")
            if len(segs) >= 4 and segs[0].isdigit() and segs[1].isdigit() and segs[2].isdigit():
                run_name = "_".join(segs[3:])
            else:
                run_name = run_part
        metrics_path = os.path.join(root, name, "metrics.json")
        config_path = os.path.join(root, name, "config.json")
        if not os.path.isfile(metrics_path) or not os.path.isfile(config_path):
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue
        if run_name not in by_run or ts > by_run[run_name][0]:
            by_run[run_name] = (ts, metrics, config, os.path.join(root, name))
    return [(rn, m, c, d) for rn, (_, m, c, d) in by_run.items()]


def _format_pct(x):
    if x is None:
        return "—"
    return f"{float(x)*100:.2f}%"


def _short_hyperparams(config: dict) -> str:
    """从 config 抽出主要超参，拼成短字符串"""
    keys = ["lr", "epochs_per_task", "batch_size", "slice_margin", "lambda_margin",
            "vae_epochs", "vae_n_fake_per_task", "lambda_adv_feat", "lambda_adv_slice",
            "lambda_slice", "lambda_feat", "focal_gamma", "importance_decay", "mix_alpha"]
    parts = []
    for k in keys:
        if k in config and config[k] is not None:
            v = config[k]
            if isinstance(v, float) and 0 < v < 1:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v}")
    return ", ".join(parts[:8]) if parts else "—"


def build_cross_table(mnist_runs, cifar100_runs):
    """构建横向对比：方法 | MNIST Class-IL | MNIST BWT | CIFAR-100 Class-IL (TAg) ↑ | CIFAR-100 BWT |"""
    mnist_by_key = {}
    for run_name, metrics, config, _ in mnist_runs:
        key = RUN_NAME_TO_CROSS_KEY.get(run_name)
        if key:
            mnist_by_key[key] = (
                metrics.get("final_avg_class_il"),
                metrics.get("bwt"),
            )
    cifar100_by_key = {}
    for run_name, metrics, config, _ in cifar100_runs:
        key = RUN_NAME_TO_CROSS_KEY.get(run_name)
        if key:
            cifar100_by_key[key] = (
                metrics.get("final_avg_class_il"),
                metrics.get("bwt"),
            )
    if "baseline" not in mnist_by_key:
        mnist_by_key["baseline"] = (MNIST_BASELINE_FALLBACK[0] / 100.0, MNIST_BASELINE_FALLBACK[1] / 100.0)

    order = ["baseline", "ewc", "si", "slice_margin", "feat_kd", "logit_kd", "slice_margin_feat_kd", "slice_margin_logit_kd", "slice_margin_ewc", "vae_si", "dual_discriminator", "stronger_replay"]
    lines = [
        "| 方法 | MNIST Class-IL ↑ | MNIST BWT | CIFAR-100 Class-IL (TAg) ↑ | CIFAR-100 BWT |",
        "|------|------------------|-----------|-----------------------------|---------------|",
    ]
    for key in order:
        if key not in CROSS_KEY_TO_DISPLAY:
            continue
        label = CROSS_KEY_TO_DISPLAY[key]
        m_ci, m_bwt = mnist_by_key.get(key, (None, None))
        c_ci, c_bwt = cifar100_by_key.get(key, (None, None))
        lines.append(f"| {label} | {_format_pct(m_ci)} | {_format_pct(m_bwt)} | {_format_pct(c_ci)} | {_format_pct(c_bwt)} |")
    return "\n".join(lines)


def build_full_table(mnist_runs, cifar100_runs):
    """全部实验 + 主要超参：数据集 | 实验名 | Class-IL | BWT | 主要超参"""
    lines = [
        "| 数据集 | 实验名 | Class-IL ↑ | BWT | 主要超参 |",
        "|--------|--------|------------|-----|----------|",
    ]
    for run_name, metrics, config, _ in sorted(mnist_runs, key=lambda x: (-(x[1].get("final_avg_class_il") or 0), x[0])):
        ci = metrics.get("final_avg_class_il")
        bwt = metrics.get("bwt")
        short = _short_hyperparams(config)
        lines.append(f"| MNIST | {run_name} | {_format_pct(ci)} | {_format_pct(bwt)} | {short} |")
    for run_name, metrics, config, _ in sorted(cifar100_runs, key=lambda x: (-(x[1].get("final_avg_class_il") or 0), x[0])):
        ci = metrics.get("final_avg_class_il")
        bwt = metrics.get("bwt")
        short = _short_hyperparams(config)
        lines.append(f"| CIFAR-100 | {run_name} | {_format_pct(ci)} | {_format_pct(bwt)} | {short} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate MNIST/CIFAR-100 results and hyperparams.")
    parser.add_argument("--update-readme", action="store_true", help="Replace README block between AGGREGATE_TABLE markers.")
    args = parser.parse_args()

    mnist_runs = _load_latest_per_run("split_mnist")
    cifar100_runs = _load_latest_per_run("split_cifar100")

    cross_md = build_cross_table(mnist_runs, cifar100_runs)
    full_md = build_full_table(mnist_runs, cifar100_runs)

    body = f"""# 实验结果汇总（自动生成）

运行 `python scripts/aggregate_results.py` 生成。数据来自 `output/split_mnist/experiments` 与 `output/split_cifar100/experiments`（每实验取最新一次 run）。

## MNIST vs CIFAR-100 横向对比

{cross_md}

## 全部实验（含主要超参）

每实验的完整 config 见对应结果目录下的 `config.json`。

{full_md}
"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"Written: {RESULTS_MD}")

    if args.update_readme:
        if not os.path.isfile(README_PATH):
            print("README.md not found, skip --update-readme")
            return
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        start_marker = "<!-- AGGREGATE_TABLE -->"
        end_marker = "<!-- END_AGGREGATE_TABLE -->"
        if start_marker not in content or end_marker not in content:
            # 在「Split CIFAR-10 实验结果」段落后插入新段
            insert = "\n\n## MNIST vs CIFAR-100 横向对比与超参\n\n"
            insert += "以下横向对比表由 `python scripts/aggregate_results.py` 生成（每数据集每方法取最新一次 run）。\n\n"
            insert += cross_md + "\n\n"
            insert += "**全部实验及主要超参**见 [docs/experiment_results.md](docs/experiment_results.md)。每实验完整超参见 `output/<数据集>/experiments/<实验目录>/config.json`。\n"
            target = "结果目录：`output/split_cifar100/experiments/`"
            if target in content:
                idx = content.find(target)
                end_of_para = content.find("\n\n", idx)
                if end_of_para == -1:
                    end_of_para = len(content)
                content = content[:end_of_para] + "\n\n---" + insert + content[end_of_para:]
            else:
                content = content + "\n\n---" + insert
        else:
            new_block = start_marker + "\n\n" + cross_md + "\n\n详见 [docs/experiment_results.md](docs/experiment_results.md)。\n\n" + end_marker
            start = content.find(start_marker)
            end = content.find(end_marker) + len(end_marker)
            content = content[:start] + new_block + content[end:]
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("README.md updated with cross-table.")
    else:
        print("Tip: use --update-readme to inject the cross-table into README.")


if __name__ == "__main__":
    main()
