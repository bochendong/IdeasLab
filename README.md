# IdeasLab 实验报告

> 实验记录由 ExperimentManager 自动维护

## 实验列表

| 时间 | 脚本 | 实验名 | Final Avg Class-IL | BWT | Forgetting | 结果目录 / Log |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-11 | split_mnist/SplitMinist.py | split_mnist_groupdiff_no_replay_fixed | 25.69% | -92.21% | T0=99.9%, T1=99.2%, T2=99.5%, T3=70.2%, T4=0.0% | [结果目录](output/split_mnist/experiments/2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed) · [train.log](output/split_mnist/experiments/2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed/train.log) |

## 无 Replay 实验方案（exp1~exp4）

**Baseline**：`split_mnist/SplitMinist.py`（已有结果：Class-IL 25.69%，BWT -92.21%）

### 实验目的

Baseline 在 Split MNIST 上出现严重灾难性遗忘。分析发现**核心问题是 slice 尺度失衡**——新任务 slice 的 logit 越来越大，旧任务 slice 相对变弱，导致预测几乎总是选到最新任务。Replay 能缓解但需要存储旧数据，我们希望在**完全不使用 Replay** 的前提下，验证哪些策略能有效缓解遗忘。

### 实验设计思路

| 实验 | 脚本 | 说明 | 验证目标 |
| --- | --- | --- | --- |
| Exp1: Stronger Reg | `split_mnist/exp2_stronger_reg.py` | 更强蒸馏正则：λ_slice=8, λ_feat=2 | 仅靠更强蒸馏能否缓解 slice 失衡 |
| Exp2: Frozen Backbone | `split_mnist/exp3_frozen_backbone.py` | Task 0 完成后冻结 backbone | 限制 backbone 漂移是否有效 |
| Exp3: EWC | `split_mnist/exp4_ewc.py` | Elastic Weight Consolidation（λ_ewc=5000） | 参数重要性惩罚能否保护旧任务 |
| Exp4: Frozen + Stronger Reg | `split_mnist/exp5_frozen_stronger_reg.py` | 组合策略：冻结 backbone + 更强正则 | 组合策略是否优于单策略 |

**批量运行：**
```bash
python split_mnist/run_all_experiments.py
```

**单独运行：**
```bash
python split_mnist/exp2_stronger_reg.py
python split_mnist/exp3_frozen_backbone.py
python split_mnist/exp4_ewc.py
python split_mnist/exp5_frozen_stronger_reg.py
```

**Colab 运行**（每完成一个自动推送到 GitHub）：
- 打开 `colab_split_mnist_experiments.ipynb`，上传到 [Colab](https://colab.research.google.com/)
- 配置 GitHub Token 后顺序执行即可


