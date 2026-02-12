# IdeasLab 实验报告

> 实验记录由 ExperimentManager 自动维护

## 环境与运行（Windows）

- **Python**：3.11（推荐），创建 venv 后安装依赖：`pip install -r requirements.txt`
- **PyTorch**：若出现 `WinError 1114`（c10.dll 加载失败），请使用 **CPU 版 2.4.1**（已写在 `requirements.txt`）：
  ```bash
  pip install torch==2.4.1+cpu torchvision==0.19.1+cpu -f https://download.pytorch.org/whl/cpu
  ```
- 运行全部实验（已跑过的会跳过）：`python split_mnist/run_all_experiments.py`

---

## SOTA 与基准（Class-IL，Split MNIST）

- **设定**：5 任务 Split MNIST（0/1, 2/3, …, 8/9），**无回放**（no replay / exemplar-free），单网络、Class-IL 评估（测试时不知任务 ID）。
- **文献中典型范围**（因协议、网络、epoch 等不同而波动）：
  - **无回放**：EWC、LwF、SI 等正则方法在类似设定下，Final Avg Class-IL 多在 **20%～40%**；仅蒸馏/正则时遗忘仍明显，旧任务准确率易跌至 40% 以下。
  - **有回放**：使用 exemplar 或生成式回放时，同一 benchmark 上可达到 **90%+**；本仓库聚焦无回放，不与回放方法直接比数字。
- **本仓库当前最佳（无回放）**：**Exp7 SI** **34.15%**（Class-IL）；Exp2 更强正则 29.37%，Exp4 EWC 29.02%；Baseline 25.69%。
- **目标**：在**不增加回放**的前提下，通过 Attention、SI、Adapter、生成式伪样本等方向，逼近或超过文献中无回放 Class-IL 的典型上界（约 30%～40%），并争取更好 BWT/遗忘指标。

---

## 一、目前进行的实验总结

### 1. 任务设定

- **数据集**：Split MNIST（5 个任务，每任务 2 类：0/1, 2/3, 4/5, 6/7, 8/9）
- **评估**：Task-IL（已知任务）、Class-IL（不知任务）、遗忘（Forgetting）、BWT（Backward Transfer）
- **共同设置**：无 Replay，设备 CUDA，lr=0.005，每任务 4 epoch，batch_size=128，cosine head

### 2. 已跑完并有 metrics 的实验

| 实验 | 脚本/配置概要 | Final Avg **Task-IL** | Final Avg **Class-IL** | BWT | 遗忘概况 | 结果目录 |
|------|----------------|------------------------|------------------------|-----|----------|----------|
| **Exp2 更强正则** | λ_slice=8, λ_feat=2，不冻结、无 EWC | **98.9%** | **29.37%** | -86.26% | 严重（T0≈99.7%, …） | `output/.../2026-02-11_13-51-48_exp2_stronger_reg` |
| **Baseline 复现 (fixed)** | `SplitMinist.py`，λ_slice=2, λ_feat=0.5 | **99.4%** | 25.69% | -92.21% | 严重（T0~T3 接近全忘） | `output/.../2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed` |
| **Exp4 EWC** | λ_slice=2, λ_feat=0.5，**λ_ewc=5000**，不冻结 | **99.2%** | **29.02%** | -87.92% | 严重（T0≈100%, …） | `output/.../2026-02-11_14-54-30_exp4_ewc` |
| **Exp5 冻结+更强正则** | λ_slice=8, λ_feat=2，**Task 0 后冻结** backbone | **82.5%** | 16.65% | -81.14% | 严重 | `output/.../2026-02-11_15-14-36_exp5_frozen_stronger_reg` |
| **Exp6 Attention Backbone** | patch+multi-head attention backbone，λ_slice=2, λ_feat=0.5 | **98.6%** | 26.27% | -90.57% | 严重 | `output/.../2026-02-11_17-14-37_exp6_attention_backbone` |
| **Exp7 SI** | Synaptic Intelligence λ_si=1，λ_slice=2, λ_feat=0.5 | **99.2%** | **34.15%** | -80.88% | 较轻 | `output/.../2026-02-11_17-33-31_exp7_si` |
| **Exp8 Attention + SI** | Attention backbone + SI | **99.1%** | 29.64% | -84.61% | 严重 | `output/.../2026-02-11_17-51-42_exp8_attention_plus_si` |

### 3. 已跑完、仅有 train.log 的实验

| 实验 | 配置概要 | Task-IL（约） | Class-IL | 结论 |
|------|----------|---------------|----------|------|
| **Exp3 冻结 Backbone** | Task 0 后冻结 backbone，仅训 head | **80%～99%** | 严重崩塌（旧任务≈0%） | 冻结后旧类在 Class-IL 下无法保持 |

### 4. 方法小结：什么有用、什么没用

- **对 Task-IL 有用（我们做得很好，多任务平均 90%+）**
  - **Slice 蒸馏 + 特征蒸馏**（Baseline / Exp2 / Exp4）：已知任务时，各任务 slice 保持良好，Task-IL 约 **98%～99%**。
  - **EWC**（Exp4）：与蒸馏一起用时，Task-IL 同样 **99%+**，说明“已知任务 ID”时模型能正确选 slice 并分类。
- **对 Class-IL 有用（有限提升）**
  - **更强正则**（Exp2：λ_slice=8, λ_feat=2）略好于 Baseline，Class-IL **29.37%** vs 25.69%。
  - **EWC**（Exp4）与 Exp2 接近（**29.02%**），未明显拉开差距。
  - **SI**（Exp7）**34.15%**，目前 Class-IL 最高，BWT -80.88% 也优于 Exp2/Exp4；**Attention backbone**（Exp6）26.27%，**Attention+SI**（Exp8）29.64%。
- **对 Class-IL 没用或有害**
  - **冻结 Backbone**（Exp3）：Class-IL 旧任务几乎归零，仅当前任务高；Task-IL 尚可，说明问题在“无任务 ID 时无法选对 slice”。
  - **冻结 + 更强正则**（Exp5）：Task-IL 掉到 **82.5%**，Class-IL **16.65%** 最低，说明过度限制 backbone 反而损害多任务表征。
- **结论**：当前设定下 **Task-IL 已接近满分，瓶颈在 Class-IL**：模型在“知道任务”时表现很好，在“不知道任务”时无法稳定地把样本路由到正确 slice，导致 Class-IL 崩塌。

### 5. 如何把 Task-IL 同步到 Class-IL

Task-IL 高、Class-IL 低，本质是**测试时没有任务 ID**，模型必须在 10 类上直接做决策，而各任务的 logit slice 之间存在**竞争与失衡**（新任务 slice 压制旧任务）。因此“把 Task-IL 同步到 Class-IL”等价于：**在无任务 ID 的前提下，仍能选对“该用哪个任务的 slice”**。可尝试方向：

1. **Slice 平衡与校准**：约束各任务 slice 的 scale/范数，避免后学任务 logits 系统性大于先学任务（当前 group-diff 日志已在观察 slice strength / gap）。
2. **任务不可知的路由**：用轻量模块根据输入预测“更可能属于哪一任务”，再加权或选择对应 slice，而不是直接对所有 slice 做 argmax（类似 task inference / Pseudo-task 思路）。
3. **统一表征 + 类边界**：减少“每任务一块 slice”的割裂，让 backbone 输出更任务无关的特征，在 10 类空间上直接学边界（难度大，易遗忘）。
4. **生成式/伪样本**（Exp9）：无真实回放前提下用 VAE 等生成旧类样本，让分类器在 10 类上持续看到旧类，有助于 Class-IL 边界不塌。
5. **Attention / Adapter**（Exp6 / Exp8 / Exp10）：通过更强表征或任务专用 adapter，让“同一 backbone 下不同任务”的 logits 更可区分、更平衡，从而在 Class-IL 下更稳。

后续实验可围绕：**slice 平衡正则、任务推理模块、以及 Exp6–10 的 Class-IL 结果**，系统对比哪条路最能将 Task-IL 的优势迁移到 Class-IL。

### 6. 新增实验方向（冲击 Class-IL SOTA，无回放）

在现有正则/EWC/冻结基础上，新增以下方向，均**不使用真实样本回放**：

- **Attention**：Exp6 / Exp8 使用 patch + multi-head self-attention 的 backbone，提升表征与任务区分。
- **Synaptic Intelligence (SI)**：Exp7 / Exp8 使用突触重要性正则，与 EWC 互补。
- **迁移/适配器**：Exp10 每任务一个 bottleneck adapter，共享 backbone，迁移学习风格。
- **生成式伪样本**：Exp9 每任务训练 CVAE，新任务时从旧任务 VAE 采样参与 CE 损失，**不存储任何真实样本**（可视为无回放或“伪回放”）。

扩散模型方向可在后续用轻量扩散替代 VAE 做生成式正则或伪样本。

---

## 已完成实验总结（有 metrics 的）

| 实验 | Final Avg Task-IL | Final Avg Class-IL | BWT | 备注 |
|------|-------------------|--------------------|-----|------|
| Baseline (fixed) | 99.4% | 25.69% | -92.21% | 基准 |
| **Exp2** 更强正则 | 98.9% | 29.37% | -86.26% | 略优于 Baseline |
| **Exp3** 冻结 Backbone | ~90% | 崩塌 | — | 仅 train.log |
| **Exp4** EWC | 99.2% | 29.02% | -87.92% | 与 Exp2 接近 |
| **Exp5** 冻结+更强正则 | 82.5% | 16.65% | -81.14% | 最差 |
| **Exp6** Attention Backbone | 98.6% | 26.27% | -90.57% | 架构增强 |
| **Exp7** SI | 99.2% | **34.15%** | -80.88% | **当前 Class-IL 最佳** |
| **Exp8** Attention + SI | 99.1% | 29.64% | -84.61% | 组合 |
| Exp9 VAE 伪样本 | — | — | — | 未跑完 / 无 metrics |
| Exp10～Exp13 | — | — | — | 待跑 |

---

## 二、实验列表（明细）

| 时间 | 脚本 | 实验名 | Final Avg **Task-IL** | Final Avg **Class-IL** | BWT | Forgetting | 结果目录 / Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-11 | split_mnist\exp2_stronger_reg.py | exp2_stronger_reg | **98.9%** | 29.37% | -86.26% | T0=99.7%, … | [结果目录](output\split_mnist\experiments\2026-02-11_13-51-48_exp2_stronger_reg) · [train.log](output\split_mnist\experiments\2026-02-11_13-51-48_exp2_stronger_reg/train.log) |
| 2026-02-11 | split_mnist/SplitMinist.py | split_mnist_groupdiff_no_replay_fixed | **99.4%** | 25.69% | -92.21% | T0=99.9%, … | [结果目录](output/split_mnist/experiments/2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed) · [train.log](output/split_mnist/experiments/2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed/train.log) |
| 2026-02-11 | split_mnist\exp3_frozen_backbone.py | exp3_frozen_backbone | **~90%** | 崩塌 | — | 旧任务≈0% | [结果目录](output/split_mnist/experiments/2026-02-11_14-12-14_exp3_frozen_backbone) · [train.log](output/split_mnist/experiments/2026-02-11_14-12-14_exp3_frozen_backbone/train.log) |
| 2026-02-11 | split_mnist\exp4_ewc.py | exp4_ewc | **99.2%** | 29.02% | -87.92% | T0=100%, … | [结果目录](output/split_mnist/experiments/2026-02-11_14-54-30_exp4_ewc) · [train.log](output/split_mnist/experiments/2026-02-11_14-54-30_exp4_ewc/train.log) |
| 2026-02-11 | split_mnist\exp5_frozen_stronger_reg.py | exp5_frozen_stronger_reg | **82.5%** | 16.65% | -81.14% | 严重 | [结果目录](output/split_mnist/experiments/2026-02-11_15-14-36_exp5_frozen_stronger_reg) · [train.log](output/split_mnist/experiments/2026-02-11_15-14-36_exp5_frozen_stronger_reg/train.log) |

## 三、实验设计（脚本与验证目标）

| 实验 | 脚本 | 说明 | 验证目标 |
| --- | --- | --- | --- |
| Exp2: Stronger Reg | `split_mnist/exp2_stronger_reg.py` | 更强蒸馏正则：λ_slice=8, λ_feat=2 | 仅靠更强蒸馏能否缓解 slice 失衡 |
| Exp3: Frozen Backbone | `split_mnist/exp3_frozen_backbone.py` | Task 0 完成后冻结 backbone | 限制 backbone 漂移是否有效 |
| Exp4: EWC | `split_mnist/exp4_ewc.py` | Elastic Weight Consolidation（λ_ewc=5000） | 参数重要性惩罚能否保护旧任务 |
| Exp5: Frozen + Stronger Reg | `split_mnist/exp5_frozen_stronger_reg.py` | 组合策略：冻结 backbone + 更强正则 | 组合策略是否优于单策略 |
| **Exp6: Attention Backbone** | `split_mnist/exp6_attention_backbone.py` | Backbone 使用 patch + multi-head self-attention | 更强表征与任务区分，冲击更高 Class-IL |
| **Exp7: SI** | `split_mnist/exp7_si.py` | Synaptic Intelligence（λ_si=1） | 突触重要性正则，与 EWC 互补 |
| **Exp8: Attention + SI** | `split_mnist/exp8_attention_plus_si.py` | Attention backbone + SI | 架构与正则协同 |
| **Exp9: VAE 伪样本** | `split_mnist/exp9_vae_pseudo_replay.py` | 每任务 CVAE 生成伪样本参与 CE，无真实回放存储 | 生成式“伪回放”能否提升 Class-IL |
| **Exp10: Adapters** | `split_mnist/exp10_adapters.py` | 每任务一个 bottleneck adapter，共享 backbone | 迁移/适配器风格，减轻灾难性遗忘 |
| **Exp11: Slice 平衡** | `split_mnist/exp11_slice_balance.py` | 各 slice 强度方差损失，避免新任务压旧任务 | 把 Task-IL 同步到 Class-IL（平衡） |
| **Exp12: Slice margin** | `split_mnist/exp12_slice_margin.py` | 正确 slice max > 其它 slice max + margin | 把 Task-IL 同步到 Class-IL（拉大 gap） |
| **Exp13: Task inference** | `split_mnist/exp13_task_inference.py` | task head 预测任务，推理时用概率加权 slice | 把 Task-IL 同步到 Class-IL（路由） |


## 实验列表

| 时间 | 脚本 | 实验名 | Final Avg Class-IL | BWT | Forgetting | 结果目录 / Log |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-11 | split_mnist\exp8_attention_plus_si.py | exp8_attention_plus_si | 29.64% | -84.61% | T0=99.9%, T1=97.0%, T2=81.6%, T3=60.0%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-51-42_exp8_attention_plus_si) · [train.log](output\split_mnist\experiments\2026-02-11_17-51-42_exp8_attention_plus_si/train.log) |
| 2026-02-11 | split_mnist\exp7_si.py | exp7_si | 34.15% | -80.88% | T0=95.6%, T1=98.1%, T2=74.3%, T3=55.5%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-33-31_exp7_si) · [train.log](output\split_mnist\experiments\2026-02-11_17-33-31_exp7_si/train.log) |
| 2026-02-11 | split_mnist\exp6_attention_backbone.py | exp6_attention_backbone | 26.27% | -90.57% | T0=100.0%, T1=98.0%, T2=98.8%, T3=65.5%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-14-37_exp6_attention_backbone) · [train.log](output\split_mnist\experiments\2026-02-11_17-14-37_exp6_attention_backbone/train.log) |

