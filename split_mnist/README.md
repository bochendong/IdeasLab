# Split MNIST 实验脚本

本目录存放 Split MNIST 方向的各类实验脚本，每个脚本对应一种方法或变体。

## 运行方式

从项目根目录运行：

```bash
python split_mnist/SplitMinist.py
```

或运行其他脚本：

```bash
python split_mnist/XXX.py
```

## 目录结构

```
split_mnist/
├── common/           # 公共模块
│   ├── constants.py  # TASK_DIR 等常量
│   ├── data.py       # 数据加载、DataLoader
│   └── eval.py       # 评估：cal_acc_class_il, cal_acc_task_il, evaluate_after_task, compute_forgetting, compute_bwt
├── base_experiment.py # 无 Replay 实验基础模块
├── SplitMinist.py    # 基础版本（Baseline）
├── exp2_stronger_reg.py
├── exp3_frozen_backbone.py
├── exp4_ewc.py
├── exp5_frozen_stronger_reg.py
└── run_all_experiments.py  # 批量运行 exp1~exp5
```

## 无 Replay 实验（exp1~exp4）

Baseline 为 `SplitMinist.py`。

| 实验 | 文件 | 说明 |
|------|------|------|
| Exp1 | `exp2_stronger_reg.py` | 更强正则: λ_slice=8, λ_feat=2 |
| Exp2 | `exp3_frozen_backbone.py` | Task 0 后冻结 backbone |
| Exp3 | `exp4_ewc.py` | EWC 惩罚（λ_ewc=5000） |
| Exp4 | `exp5_frozen_stronger_reg.py` | Frozen + 更强正则 |

批量运行：`python split_mnist/run_all_experiments.py`

## 已有脚本

- `SplitMinist.py` - 基础版本：Group-Difference Logging，无 replay
