# Split CIFAR-100（10×10）

在 CIFAR-100 上做增量学习实验，**评估结构与 AdaGauss 一致**：TAw（Task-Aware）、TAg（Task-Agnostic / Class-IL）、acc 矩阵、avg_accs、forgetting、BWT。便于与 `output/adagauss_results/` 和 `scripts/plot_adagauss_results.py` 对齐。

## 设置

- **任务划分**：10 个任务，每任务 10 类，与 AdaGauss 的 `cifar100_icarl` 使用相同 `class_order`。
- **Backbone**：默认 **ResNet18**（CIFAR 32×32，特征 64 维），与 AdaGauss 的 `nnet=resnet18`、`S=64` 一致，便于公平对比。可选 `--backbone cnn` 使用轻量 CNN 快速试跑。
- **数据路径**：`data/cifar100_100`（首次运行会自动下载 CIFAR-100）。
- **评估**：每学完一个任务，在所有已学任务上计算 TAw / TAg，得到与 AdaGauss 相同的 `acc_taw`、`acc_tag`、`avg_accs_taw`、`avg_accs_tag`，并写入 `results/`，可用 `python scripts/plot_adagauss_results.py --results-dir output/split_cifar100/experiments/<实验目录>` 画图。

## 运行

```bash
# 项目根目录
python split_cifar100/run_baseline.py              # 单实验：基线
python split_cifar100/run_baseline.py --epochs 5   # 快速试跑
python split_cifar100/run_baseline.py --backbone cnn --epochs 2  # 轻量 CNN 快速验证

# 跑全部组合实验（slice margin、蒸馏、EWC、SI 等），并保留记录
python split_cifar100/run_all_experiments.py
python split_cifar100/run_all_experiments.py --epochs 5   # 快速试跑
python split_cifar100/run_all_experiments.py --list       # 列出实验名
```
实验列表与记录见 [docs/cifar100_experiments.md](../docs/cifar100_experiments.md)；聚合到 MNIST vs CIFAR-100 表见 `python scripts/aggregate_results.py`。

结果目录：`output/split_cifar100/experiments/<timestamp>_cifar100_baseline/`，内含：

- `config.json`、`metrics.json`、`train.log`
- `results/acc_taw-*.txt`、`results/acc_tag-*.txt`、`results/avg_accs_*.txt`（AdaGauss 格式）

## 与 MNIST 思路的对应

- 多任务 head（每任务一个 10 类头）、CE 训练、可选后续加 EWC/SI/slice 正则等，与 `split_mnist/base_experiment.py` 思路一致。
- 评估指标与 AdaGauss 一致，便于在 CIFAR-100 上对比和画曲线。
