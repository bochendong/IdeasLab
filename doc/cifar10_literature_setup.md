# Split CIFAR-10 与文献设定对比

本仓库当前在 Split CIFAR-10（5 任务 × 2 类）上 Class-IL 约 20%～26%，与「文献里 ResNet-18 无回放约 60%」差距大。以下对比**原论文/基准代码**的设定，便于对齐复现。

---

## 1. 重要澄清：60% 是 CIFAR-100，不是 CIFAR-10

- **AdaGauss (NeurIPS 2024)** 报告的 **60.2%** 是在 **CIFAR-100**（10 任务 × 10 类）上的结果，backbone 为 ResNet-18。  
- **CIFAR-10**（5 任务 × 2 类）与 CIFAR-100 的难度、任务数、每任务类别数都不同，不能直接拿 60% 作为 CIFAR-10 的目标值。  
- 若要对齐文献，需区分配置是 **CIFAR-10** 还是 **CIFAR-100**。

---

## 2. 原论文/基准代码中的典型设定

### 2.1 FACIL（TPAMI 基准）

- **代码**: [mmasana/FACIL](https://github.com/mmasana/FACIL)
- **默认训练**（`main_incremental.py`）:
  - **nepochs**: **200**（每任务）
  - **lr**: **0.1**
  - **优化器**: 未在摘要里写死，常见为 **SGD**
  - **batch-size**: 64（默认）
  - **网络**: ResNet-32（CIFAR 用 32 层 ResNet，非 torchvision ResNet-18）
- **数据集**: 脚本以 CIFAR-100（10 任务）为主，非 5 任务 CIFAR-10。

### 2.2 AdaGauss（基于 FACIL）

- **代码**: [grypesc/AdaGauss](https://github.com/grypesc/AdaGauss)，基于 FACIL
- **脚本** `scripts/cifar-10x10.sh`（CIFAR-100，10 任务 × 10 类）:
  - **nepochs**: **200**
  - **lr**: **0.1**
  - **batch-size**: 256
  - **weight-decay**: 5e-4
  - **优化器**: 一般为 SGD（与 FACIL 一致）
  - **数据集**: `cifar100_icarl`，**nc-first-task 10**（10×10）

---

## 3. 本仓库当前设定 vs 文献

| 项目         | 本仓库（当前）     | FACIL / AdaGauss 典型 |
|--------------|--------------------|------------------------|
| 每任务 epoch | **12**             | **200**                |
| 学习率       | **3e-4** (Adam)    | **0.1** (SGD)          |
| 优化器       | Adam               | SGD (+ momentum)       |
| weight decay | 0                  | 5e-4                   |
| batch size   | 128                | 64～256                |
| 数据增强     | RandomCrop+Flip+Norm | 与 CIFAR 常用一致    |
| 头结构       | cosine head (10 类) | 线性头 / 多 head     |

**结论**：差异主要来自 **每任务训练量（12 vs 200 epoch）** 和 **优化器/学习率（Adam 3e-4 vs SGD 0.1）**。要靠近文献里的曲线，建议至少尝试「文献式」配置：**更多 epoch（如 100～200）+ SGD lr=0.1 + weight_decay=5e-4**。

---

## 4. 如何用本仓库跑「文献式」配置

- 使用 **无 slice/feat 正则** 的 baseline（CE + cosine head），便于和「纯 fine-tuning」对比。  
- 在 `base_experiment.py` 中已支持通过 `config` 传入：
  - `optimizer`: `"adam"`（默认）或 `"sgd"`
  - `epochs_per_task`、`lr`、`weight_decay`、`sgd_momentum`  
- 可运行 **文献式 baseline**（SGD、多 epoch、无正则）:
  ```bash
  python split_cifar10/exp_baseline_facil_style.py
  ```
  或通过 `run_cifar10_experiments.py` 的 `baseline_facil_style` 条目（若已添加）。  
- 建议先跑该配置看 Class-IL；若仍明显低于预期，再对照 FACIL/AdaGauss 的 **数据划分、评测协议（TAw/TAg）、是否 fix BN** 等逐项对齐。

---

## 5. 为何无回放时 Class-IL 容易只有 20%～30%

- **单头 + 无回放**：每个新任务只在新类上做 CE，梯度会**改写 backbone 和整头**，旧类对应的 head 行虽不直接更新，但 backbone 一变，旧类特征就漂移，等价于遗忘。
- 文献里 50%～60% 多为 **CIFAR-100** 或 **有回放/强正则/专用结构**（如 AdaGauss、PRL）。**5 任务 Split CIFAR-10、严格无回放**下，很多工作本身也只报 20%～40% 量级。
- 可尝试的**不改结构的缓解**：
  - **逐任务衰减学习率**：`lr_per_task_decay=0.5`（task t 用 `lr * 0.5^t`），减轻对新任务过度拟合、对旧任务覆盖。
  - **FACIL 风格**：SGD、多 epoch、weight decay（已提供 `baseline_facil_style`）。
  - 若需明显提升：**加少量真实回放**（exemplar）或采用带回放/原型的方法，再与本仓库无回放结果对比。

---

## 6. 参考链接

- FACIL: https://github.com/mmasana/FACIL  
- AdaGauss: https://github.com/grypesc/AdaGauss  
- FACIL 论文（TPAMI）: [Class-incremental learning: survey and performance evaluation](https://arxiv.org/abs/2010.15277)  
- AdaGauss 论文（NeurIPS 2024）: [Task-recency bias strikes back...](https://arxiv.org/abs/2409.18265)
