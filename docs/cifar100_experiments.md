# Split CIFAR-100 实验记录

实验在 `output/split_cifar100/experiments/` 下，每实验一个目录：`<timestamp>_<run_name>/`，内含 `config.json`、`metrics.json`、`train.log`、`results/acc_taw-*.txt` 等。  
运行 `python scripts/aggregate_results.py` 会汇总到 [experiment_results.md](experiment_results.md)（MNIST vs CIFAR-100 表）。

## 实验列表（run_all_experiments.py 注册表）

| run_name | 说明 |
|----------|------|
| `cifar100_baseline` | ResNet18 + 10 head，仅 CE |
| `cifar100_slice_margin` | Slice margin：正确任务 head 的 max > 其它 + margin |
| `cifar100_feat_kd` | 特征蒸馏 MSE(new_feat, old_feat) |
| `cifar100_logit_kd` | Logit 蒸馏（LwF）对旧 head |
| `cifar100_slice_margin_feat_kd` | Slice margin + 特征蒸馏 |
| `cifar100_slice_margin_logit_kd` | Slice margin + logit 蒸馏 |
| `cifar100_ewc` | EWC |
| `cifar100_si` | Synaptic Intelligence |
| `cifar100_slice_margin_ewc` | Slice margin + EWC |

## 运行方式

```bash
# 跑全部
python split_cifar100/run_all_experiments.py

# 快速试跑（少 epoch）
python split_cifar100/run_all_experiments.py --epochs 5

# 只跑指定实验
python split_cifar100/run_all_experiments.py --experiments cifar100_baseline cifar100_slice_margin

# 列出所有实验名
python split_cifar100/run_all_experiments.py --list
```

## 结果汇总（需跑完后由 aggregate_results 或本表手动更新）

| 实验 | Class-IL (TAg) ↑ | BWT | 备注 |
|------|------------------|-----|------|
| cifar100_baseline | — | — | 基线 |
| cifar100_slice_margin | — | — | |
| cifar100_feat_kd | — | — | |
| cifar100_logit_kd | — | — | |
| cifar100_slice_margin_feat_kd | — | — | |
| cifar100_slice_margin_logit_kd | — | — | |
| cifar100_ewc | — | — | |
| cifar100_si | — | — | |
| cifar100_slice_margin_ewc | — | — | |

数值来源：各实验目录下 `metrics.json` 的 `final_avg_class_il`、`bwt`。运行 `python scripts/aggregate_results.py` 后，CIFAR-100 列会出现在 [experiment_results.md](experiment_results.md) 的「全部实验」表中。
