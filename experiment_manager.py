# ===== Experiment Manager - 训练实验管理与日志 =====
"""
每次训练完成后自动：
1. 将结果保存到带时间戳的文件夹
2. 更新 README.md 作为实验报告
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# 项目根目录（experiment_manager 所在目录）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "output")
README_PATH = os.path.join(_PROJECT_ROOT, "README.md")


def _get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_output_dir(experiment_group: Optional[str] = None) -> str:
    """获取该实验组的输出根目录。如 'split_mnist' -> output/split_mnist/"""
    if experiment_group:
        return os.path.join(OUTPUT_ROOT, experiment_group)
    return OUTPUT_ROOT


def _get_experiments_root(experiment_group: Optional[str] = None) -> str:
    """获取实验根目录。experiment_group 如 'split_mnist'，则 output/split_mnist/experiments/"""
    if experiment_group:
        return os.path.join(OUTPUT_ROOT, experiment_group, "experiments")
    return os.path.join(OUTPUT_ROOT, "experiments")


def create_experiment_dir(run_name: str, experiment_group: Optional[str] = None) -> str:
    """创建本次实验的目录，返回绝对路径"""
    experiments_root = _get_experiments_root(experiment_group)
    os.makedirs(experiments_root, exist_ok=True)
    ts = _get_timestamp()
    dir_name = f"{ts}_{run_name}"
    exp_dir = os.path.join(experiments_root, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config(exp_dir: str, config: Dict[str, Any]) -> None:
    """保存训练配置到 config.json"""
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_metrics(exp_dir: str, metrics: Dict[str, Any]) -> None:
    """保存指标到 metrics.json"""
    path = os.path.join(exp_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def copy_log_to_experiment(exp_dir: str, log_path: str) -> None:
    """将训练日志复制到实验目录（若已在实验目录内则跳过）"""
    if not log_path or not os.path.exists(log_path):
        return
    dst = os.path.join(exp_dir, "train.log")
    if os.path.normpath(os.path.abspath(log_path)) == os.path.normpath(os.path.abspath(dst)):
        return  # 已在实验目录内，无需复制
    import shutil
    shutil.copy2(log_path, dst)


def save_model(exp_dir: str, model, name: str = "model.pt") -> None:
    """保存模型 checkpoint"""
    try:
        import torch
        path = os.path.join(exp_dir, name)
        torch.save(model.state_dict(), path)
    except Exception as e:
        print(f"[ExperimentManager] 保存模型失败: {e}")


def _ensure_readme_exists() -> None:
    """确保 README 存在，若不存在则创建模板"""
    if not os.path.exists(README_PATH):
        content = """# IdeasLab 实验报告

> 实验记录由 ExperimentManager 自动维护

## 实验列表

"""
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(content)


def _parse_readme_table(readme_content: str) -> Tuple[str, list]:
    """
    解析 README，返回 (header 含 ## 实验列表, 现有数据行列表)
    """
    marker = "## 实验列表"
    if marker not in readme_content:
        return readme_content + "\n\n" + marker + "\n\n", []

    parts = readme_content.split(marker, 1)
    header = parts[0] + "## 实验列表\n\n"
    rest = parts[1] if len(parts) > 1 else ""

    # 解析表格：找 | xxx | 格式的数据行（排除表头、分隔行、表头行）
    data_rows = []
    skip_first_cells = ("时间", "脚本")  # 表头行的首列
    for line in rest.split("\n"):
        s = line.strip()
        if not s.startswith("|") or "---" in s or s == "|":
            continue
        first_cell = s.split("|")[1].strip() if "|" in s else ""
        if first_cell in skip_first_cells:
            continue  # 跳过表头行
        data_rows.append(line)
    return header, data_rows


def _format_experiment_row(
    run_name: str,
    exp_dir: str,
    timestamp: str,
    script_file: str,
    config: Dict,
    metrics: Dict,
) -> str:
    """生成单行实验记录（Markdown 表格行）"""
    rel_dir = os.path.relpath(exp_dir, _PROJECT_ROOT)
    final_acc = metrics.get("final_avg_class_il", 0) * 100
    bwt = metrics.get("bwt", 0) * 100
    forget = metrics.get("forgetting", {})
    keys = sorted(forget, key=lambda x: int(x) if str(x).isdigit() else x)
    forget_str = ", ".join([f"T{k}={float(forget.get(k, forget.get(str(k), 0)))*100:.1f}%" for k in keys])

    return (
        f"| {timestamp} | {script_file} | {run_name} | {final_acc:.2f}% | {bwt:.2f}% | {forget_str} | "
        f"[结果目录]({rel_dir}) · [train.log]({rel_dir}/train.log) |"
    )


def update_readme(
    run_name: str,
    exp_dir: str,
    script_file: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """
    更新 README.md，在实验列表表格中追加新实验记录。
    """
    _ensure_readme_exists()

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    header, data_rows = _parse_readme_table(content)
    exp_basename = os.path.basename(exp_dir)
    timestamp = exp_basename.split("_", 1)[0] if "_" in exp_basename else _get_timestamp()

    # 表头
    table_header = "| 时间 | 脚本 | 实验名 | Final Avg Class-IL | BWT | Forgetting | 结果目录 / Log |\n"
    table_header += "| --- | --- | --- | --- | --- | --- | --- |\n"

    new_row = _format_experiment_row(run_name, exp_dir, timestamp, script_file, config, metrics)
    new_lines = [new_row] + data_rows
    new_table = table_header + "\n".join(new_lines) + "\n\n"

    new_content = header + new_table

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)


class ExperimentManager:
    """
    训练实验管理器。
    用法：
        with ExperimentManager("my_run", config_dict, script_file=__file__) as mgr:
            # 训练...
            mgr.finish(metrics_dict)

    每个实验会：
    - 创建 output/实验组/experiments/时间戳_实验名/ 文件夹
    - experiment_group 如 'split_mnist'，则 output/split_mnist/experiments/
    - 保存 config.json, metrics.json, train.log
    - 在 README 中记录：脚本、实验名、指标、结果目录、log 链接
    """

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        script_file: Optional[str] = None,
        experiment_group: Optional[str] = None,
    ):
        self.run_name = run_name
        self.experiment_group = experiment_group  # 如 'split_mnist'，输出到 output/split_mnist/experiments/
        self.config = dict(config) if config else {}
        # 脚本路径：相对于项目根显示，如 split_mnist/SplitMinist.py
        if script_file:
            script_abs = os.path.abspath(script_file)
            if script_abs.startswith(_PROJECT_ROOT):
                self.script_file = os.path.relpath(script_abs, _PROJECT_ROOT)
            else:
                self.script_file = os.path.basename(script_file)
        else:
            self.script_file = "unknown"
        self.config["script_file"] = self.script_file
        if experiment_group:
            self.config["experiment_group"] = experiment_group
        self.exp_dir: Optional[str] = None
        self.log_path: Optional[str] = None

    def __enter__(self) -> "ExperimentManager":
        self.exp_dir = create_experiment_dir(self.run_name, self.experiment_group)
        save_config(self.exp_dir, self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 即使异常也尝试保存已有内容
        pass

    def set_log_path(self, path: str) -> None:
        self.log_path = path

    def finish(
        self,
        metrics: Dict[str, Any],
        model=None,
        save_model_checkpoint: bool = False,
    ) -> None:
        """训练完成后调用，保存所有结果并更新 README"""
        if not self.exp_dir:
            return

        save_metrics(self.exp_dir, metrics)

        if self.log_path:
            copy_log_to_experiment(self.exp_dir, self.log_path)

        if save_model_checkpoint and model is not None:
            save_model(self.exp_dir, model)

        update_readme(self.run_name, self.exp_dir, self.script_file, self.config, metrics)
        print(f"[ExperimentManager] 结果已保存到: {self.exp_dir}")
        print(f"[ExperimentManager] README 已更新")
