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
- **本仓库当前最佳（无回放）**：**Exp9 VAE 伪样本** **40.35%**（Class-IL），BWT -71.37%；其次 Exp7 SI 34.15%、Exp2 29.37%、Exp4 EWC 29.02%；Baseline 25.69%。
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
| **Baseline** | `SplitMinist.py`，λ_slice=2, λ_feat=0.5 | **99.4%** | 25.69% | -92.21% | 严重（T0~T3 接近全忘） | `output/.../2026-02-11_15-44-06_split_mnist_groupdiff_no_replay_fixed` |
| **Exp4 EWC** | λ_slice=2, λ_feat=0.5，**λ_ewc=5000**，不冻结 | **99.2%** | **29.02%** | -87.92% | 严重（T0≈100%, …） | `output/.../2026-02-11_14-54-30_exp4_ewc` |
| **Exp5 冻结+更强正则** | λ_slice=8, λ_feat=2，**Task 0 后冻结** backbone | **82.5%** | 16.65% | -81.14% | 严重 | `output/.../2026-02-11_15-14-36_exp5_frozen_stronger_reg` |
| **Exp6 Attention Backbone** | patch+multi-head attention backbone，λ_slice=2, λ_feat=0.5 | **98.6%** | 26.27% | -90.57% | 严重 | `output/.../2026-02-11_17-14-37_exp6_attention_backbone` |
| **Exp7 SI** | Synaptic Intelligence λ_si=1，λ_slice=2, λ_feat=0.5 | **99.2%** | **34.15%** | -80.88% | 较轻 | `output/.../2026-02-11_17-33-31_exp7_si` |
| **Exp8 Attention + SI** | Attention backbone + SI | **99.1%** | 29.64% | -84.61% | 严重 | `output/.../2026-02-11_17-51-42_exp8_attention_plus_si` |
| **Exp9 VAE 伪样本** | 每任务 CVAE 生成伪样本参与 CE，无真实回放 | **97.1%** | **40.35%** | -71.37% | 较轻（T0~T3 约 63%～81%） | `output/.../2026-02-11_18-10-46_exp9_vae_pseudo_replay` |
| **Exp12 Slice margin** | λ_slice_margin=1, margin=0.5，正确 slice max > 其它 + margin | **99.3%** | **26.31%** | -91.30% | 严重（T0≈99.9%, T1≈99.0%, T2≈95.6%, T3≈70.7%, T4=0%） | `output/.../2026-02-11_19-06-57_exp12_slice_margin` |
| **Exp13 Task inference** | task head 预测任务、推理时用概率加权 slice，λ_task_inference=1 | **77.4%** | **19.81%** | -97.90% | 极严重（T0～T3 几乎全忘，task head 偏向预测最新任务） | `output/.../2026-02-11_19-25-19_exp13_task_inference` |
| **Exp14 特征空间原型增强** | PASS 风：类均值 + 高斯噪声伪特征参与 CE，proto_aug_r=0.4, n=8/类 | **99.4%** | **27.69%** | -89.52% | 严重（T0≈99.9%, …, T3≈66%） | `output/.../2026-02-11_20-14-56_exp14_proto_aug` |
| **Exp15 PRL 预留空间** | 顶会 PRL 风：Base 收紧/预留空间，新任务推入预留区 | **99.4%** | **29.00%** | -88.00% | 严重（T0≈99.9%, …, T3≈60%） | `output/.../2026-02-11_20-33-22_exp15_prl_base_reserve` |
| **Exp16 PASS SSL** | 原型增强 + 自监督（PASS 风） | **98.5%** | **25.90%** | -91.12% | 严重 | `output/.../2026-02-11_20-51-25_exp16_pass_ssl` |
| **Exp17 LDC 漂移补偿** | 可学习 projector 把特征映射回旧空间，旧类用投影特征与旧原型匹配 | **99.4%** | **26.89%** | -90.61% | 严重（T0≈99.9%, …, T3≈69%） | `output/.../2026-02-11_21-09-31_exp17_ldc_drift` |
| **Exp18 非对称 CE** | EFC 风：旧类 vs 新类非对称交叉熵，减轻新类主导 | **99.3%** | **23.33%** | -95.23% | 严重（遗忘略重） | `output/.../2026-02-11_21-27-34_exp18_asymmetric_ce` |
| **Exp19 原型增强 + SI** | Exp14 特征空间原型增强 + λ_si=1 | **99.3%** | **39.52%** | **-73.34%** | 较轻（T0～T3 约 57%～83%） | `output/.../2026-02-11_21-45-37_exp19_proto_aug_si` |

### 3. 已跑完、仅有 train.log 的实验

| 实验 | 配置概要 | Task-IL（约） | Class-IL | 结论 |
|------|----------|---------------|----------|------|
| **Exp3 冻结 Backbone** | Task 0 后冻结 backbone，仅训 head | **80%～99%** | 严重崩塌（旧任务≈0%） | 冻结后旧类在 Class-IL 下无法保持 |

### 4. 方法小结：什么有用、什么没用

**Task-IL（已知任务 ID）**  
在“给出任务”的设定下，蒸馏 + 共享 head 已经足够：Baseline / Exp2 / Exp4 / Exp6～Exp12 的 Task-IL 多在 **97%～99%**。各任务的 logit slice 在已知任务时都能被正确选用，**瓶颈不在 Task-IL**。

**Class-IL（不知任务 ID）——按效果排序**

| 梯队 | 实验 | Class-IL | BWT | 说明 |
|------|------|----------|-----|------|
| 最佳 | **Exp9 VAE 伪样本**、**Exp19 原型增强+SI** | **40.35%**、**39.52%** | -71.37%、**-73.34%** | Exp9：CVAE 图像伪样本；Exp19：特征空间原型增强 + SI，轻量且接近 Exp9。 |
| 次佳 | **Exp7 SI** | **34.15%** | -80.88% | 突触重要性正则，与 EWC 互补。 |
| 有效 | **Exp15 PRL 预留空间**、Exp2、Exp4、Exp8 | **29.00%**、29%～29.6% | -88%、-85%～-88% | Exp15：Base 预留空间 + 新任务推入预留区；其余为蒸馏/正则。 |
| 略优 | **Exp14 原型增强**、**Exp17 LDC**、Exp6、Exp12 | **27.69%**、**26.89%**、26.27%、26.31% | -89.5%、-90.6%、-90%～-91% | Exp14：仅特征空间原型增强；Exp17：漂移补偿 projector。 |
| 持平 | **Exp16 PASS SSL**、Baseline、Exp11 | **25.90%**、25.69%、25.72% | -91.1%、-92.2%、-90.6% | Exp16：原型增强+自监督，与 Baseline 接近。 |
| 更差 | **Exp18 非对称 CE**、Exp10、Exp13 | **23.33%**、20.51%、19.81% | -95.23%、-98.61%、-97.90% | Exp18：非对称 CE 在本设定下遗忘加重；Exp10/13 见前。 |
| 崩塌 | Exp3 冻结、Exp5 冻结+强正则 | ～0% / 16.65% | — | 冻结 backbone 在无任务 ID 下无法保持旧类。 |

**单点结论**

- **Exp10 Adapters**：每任务 adapter、推理时 mean 融合 → Class-IL 劣于 Baseline，当前设定下“任务专用 adapter”未缓解 slice 竞争。
- **Exp11 Slice 平衡**：约束各 slice 强度方差（λ=0.5）→ 与 Baseline 同量级，单纯平衡方差收益有限。
- **Exp12 Slice margin**：正确 slice max > 其它 + margin（λ=1, margin=0.5）→ 与 Baseline/Exp11 同量级，单纯拉大 margin 收益有限。
- **Exp13 Task inference**：轻量 task head 预测任务 ID，推理时用其概率加权各 slice。Class-IL **19.81%**、BWT **-97.90%**，**劣于 Baseline**；Task-IL 也降至 **77.4%**。group-diff 显示旧类样本几乎全被判成“最新任务”（winner_task_dist 中 W4=100%），task head 严重偏向新任务，路由失效。
- **Attention（Exp6/Exp8）**：在本设定下对 Class-IL 贡献不明显，Exp8 甚至弱于单用 SI；可能与任务数少、数据简单有关。

**与文献对照**（详见 `doc/SOTA.md`）  
无回放 Class-IL 在文献中（CIFAR-100 / ImageNet 等）典型在 **20%～40%** 区间，强方法（AdaGauss、DS-AL、VAE/伪样本等）可达 60%+ 或与回放可比。本仓库 **Exp9（40.35%）** 落在该区间上界附近，与“生成式伪样本 / 原型增强”类思路（如 PASS、AdaGauss 的类高斯建模）一致；**SI/蒸馏** 与顶会中正则/重要性加权思路一致，但单靠 slice 平衡/margin（Exp11/12）尚未带来可辨提升。

**新实验（Exp14～Exp19）总结**  
- **Exp19 原型增强+SI**：Class-IL **39.52%**、BWT **-73.34%**，与 **Exp9（40.35%）** 同档，且只存类均值+噪声、无需 VAE，**推荐作为轻量无回放方案**。  
- **Exp15 PRL 预留空间**：29.00%、BWT -88%，优于 Baseline 与多数单点方法，说明“为未来类预留空间”有效。  
- **Exp14 仅原型增强**（无 SI）：27.69%，优于 Baseline，但明显弱于 Exp19，说明**原型增强与 SI 叠加收益大**。  
- **Exp17 LDC 漂移补偿**：26.89%，略优于 Baseline，projector 补偿有一定作用但有限。  
- **Exp16 PASS SSL**：25.90%，与 Baseline 持平，当前自监督设计未带来可辨提升。  
- **Exp18 非对称 CE**：23.33%、BWT -95.23%，**劣于 Baseline**，当前非对称权重或实现未缓解新类主导。

**核心结论**  
瓶颈在 **Class-IL**：测试时无任务 ID，各 slice 竞争失衡。无真实回放下，**Exp9 VAE 伪样本** 与 **Exp19 原型增强+SI** 最佳；**SI（Exp7）**、**Exp15 PRL** 次之；**Exp14 原型增强**、**Exp17 LDC** 有有限提升；**Exp16 PASS SSL** 持平 Baseline；**Exp18 非对称 CE**、**Adapter / Task inference（Exp10/13）** 未达预期或更差；**Attention** 在本实验中贡献有限。

---

### 5. 基于顶会论文的新点子（可做后续实验）

下面结合 `doc/SOTA.md` 中的 **NeurIPS / ICLR / CVPR / ECCV / AAAI** 无回放 Class-IL 工作，提炼可落地的方向，供后续实验参考。

| 顶会思路 | 可做的新实验 | 与本仓库的衔接 |
|----------|----------------|----------------|
| **PASS (CVPR’21)**：原型增强 + 自监督 | **特征空间原型增强**：只存每类均值特征（或 slice 原型），对新任务训练时对旧类原型加高斯噪声生成“伪旧特征”，参与蒸馏或 CE。比 Exp9 的 VAE 更轻、无需生成图像。 | 已有 slice/group 统计；可加 `proto_aug` 噪声采样与损失。 |
| **AdaGauss (NeurIPS’24)**：类为高斯、自适应协方差 + anti-collapse | **Slice 高斯化 + anti-collapse**：把每个任务的 slice 视为高斯（均值 + 对角/低秩协方差），增量时更新协方差；加 anti-collapse 损失（如约束特征维度不塌缩）。 | 已有 cosine head 与 slice；可改为 NCM/高斯分类器 + 协方差更新与正则。 |
| **DS-AL / G-ACIL (AAAI/NeurIPS’24)**：解析闭式解、双流补偿 | **解析头 + 补偿分支**：旧类用递归最小二乘/闭式线性分类器（不梯度更新），新类用现有 head；或加一个小型“补偿 MLP”把特征映射到与旧类兼容的空间。 | 可先做“旧类 NCM、新类 CE”的混合头，或 tiny 补偿网络。 |
| **PRL (NeurIPS’24)**：前瞻表示、为未来类预留空间 | **Base 阶段预留空间**：Task 0 训练时加正则，让特征分布“收紧”或留出空白区域；新任务时约束新类特征落入预留区，减少对旧类区域的侵占。 | 可在 Task 0 加 spread/radius 约束，新任务加“推入预留区”的损失。 |
| **LDC (ECCV’24)**：可学习漂移补偿 | **漂移补偿模块**：backbone 每任务更新后，旧类原型在“新特征空间”里会漂移；学一个小型 projector（如 1 层 MLP）把当前特征映射到“旧空间”，使旧类原型仍有效。 | 每任务存旧类原型；新任务时加 projector，旧类用投影后特征与旧原型匹配。 |
| **TASS (CVPR’24)**：任务自适应显著性 | **显著性/注意力一致性**：对旧类样本（或伪样本）约束其 attention map / 梯度显著性 与 该任务学习时保存的模板 一致，减缓显著性漂移。 | 已有 Attention backbone（Exp6/8）；可存每任务平均 attention，加一致性损失。 |
| **EFC (ICLR’24)**：EFM + 高斯原型 + 非对称 CE | **经验特征矩阵 + 非对称 CE**：用旧类特征的二阶统计（或低秩近似）正则当前 backbone 的更新方向；对“旧类 vs 新类”使用非对称交叉熵，减轻新类主导。 | 可维护旧类特征的协方差或 EFM，加正则项；CE 对旧类降权或非对称。 |
| **ACMap / CL-LoRA (CVPR’25)**：适配器合并 + 质心原型；双适配器 | **共享 adapter + 合并**：不做“每任务一 adapter + mean”，改为 **一个共享 adapter** + 每任务只训 head；或每任务 adapter 在学完后**合并**到共享 adapter（ACMap 式），保持单模型推理。 | Exp10 的“每任务 adapter”失败；可试 shared adapter + task-specific head only，或 adapter merging。 |

**优先可试的三条**

1. **特征空间原型增强（PASS 风）**：只存每类均值特征，加噪声生成伪旧特征做蒸馏/CE，实现成本低、与 Exp9 对比清晰。  
2. **漂移补偿（LDC 风）**：固定旧类原型，新任务时学一个小 projector 把当前特征映射到“旧空间”，减轻 backbone 更新对旧类的影响。  
3. **解析头 / 旧类 NCM（DS-AL 风）**：旧类用最近类均值或闭式解，新类用 CE；减少对旧类参数的梯度更新，降低遗忘。

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
| **Exp7** SI | 99.2% | 34.15% | -80.88% | 次佳 |
| **Exp8** Attention + SI | 99.1% | 29.64% | -84.61% | 组合 |
| **Exp9** VAE 伪样本 | 97.1% | **40.35%** | -71.37% | **当前 Class-IL 最佳** |
| **Exp10** Adapters | 99.4% | 20.51% | -98.61% | 比 Baseline 更差 |
| **Exp11** Slice 平衡 | 99.1% | 25.72% | -90.56% | 与 Baseline 接近 |
| **Exp12** Slice margin | 99.3% | 26.31% | -91.30% | 与 Baseline/Exp11 接近 |
| **Exp13** Task inference | 77.4% | 19.81% | -97.90% | task head 偏向新任务，路由失效；劣于 Baseline |
| **Exp14** 原型增强 | 99.4% | 27.69% | -89.52% | 特征空间原型增强（PASS 风） |
| **Exp15** PRL 预留空间 | 99.4% | 29.00% | -88.00% | Base 预留空间 + 新任务推入预留区 |
| **Exp16** PASS SSL | 98.5% | 25.90% | -91.12% | 原型增强 + 自监督，与 Baseline 接近 |
| **Exp17** LDC 漂移补偿 | 99.4% | 26.89% | -90.61% | 可学习 projector 补偿特征漂移 |
| **Exp18** 非对称 CE | 99.3% | 23.33% | -95.23% | 非对称 CE 在本设定下更差 |
| **Exp19** 原型增强+SI | 99.3% | **39.52%** | **-73.34%** | **与 Exp9 同档，轻量无 VAE** |

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
| 2026-02-11 | split_mnist\exp19_proto_aug_si.py | exp19_proto_aug_si | 39.52% | -73.34% | T0=82.6%, T1=91.5%, T2=62.6%, T3=56.6%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_21-45-37_exp19_proto_aug_si) · [train.log](output\split_mnist\experiments\2026-02-11_21-45-37_exp19_proto_aug_si/train.log) |
| 2026-02-11 | split_mnist\exp18_asymmetric_ce.py | exp18_asymmetric_ce | 23.33% | -95.23% | T0=99.9%, T1=98.9%, T2=97.5%, T3=84.6%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_21-27-34_exp18_asymmetric_ce) · [train.log](output\split_mnist\experiments\2026-02-11_21-27-34_exp18_asymmetric_ce/train.log) |
| 2026-02-11 | split_mnist\exp17_ldc_drift.py | exp17_ldc_drift | 26.89% | -90.61% | T0=100.0%, T1=99.2%, T2=94.2%, T3=69.1%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_21-09-31_exp17_ldc_drift) · [train.log](output\split_mnist\experiments\2026-02-11_21-09-31_exp17_ldc_drift/train.log) |
| 2026-02-11 | split_mnist\exp16_pass_ssl.py | exp16_pass_ssl | 25.90% | -91.12% | T0=99.9%, T1=98.1%, T2=98.7%, T3=67.7%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_20-51-25_exp16_pass_ssl) · [train.log](output\split_mnist\experiments\2026-02-11_20-51-25_exp16_pass_ssl/train.log) |
| 2026-02-11 | split_mnist\exp15_prl_base_reserve.py | exp15_prl_base_reserve | 29.00% | -88.00% | T0=100.0%, T1=99.1%, T2=92.7%, T3=60.2%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_20-33-22_exp15_prl_base_reserve) · [train.log](output\split_mnist\experiments\2026-02-11_20-33-22_exp15_prl_base_reserve/train.log) |
| 2026-02-11 | split_mnist\exp14_proto_aug.py | exp14_proto_aug | 27.69% | -89.52% | T0=99.9%, T1=98.9%, T2=93.3%, T3=66.0%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_20-14-56_exp14_proto_aug) · [train.log](output\split_mnist\experiments\2026-02-11_20-14-56_exp14_proto_aug/train.log) |
| 2026-02-11 | split_mnist\exp13_task_inference.py | exp13_task_inference | 19.81% | -97.90% | T0=100.0%, T1=92.3%, T2=99.7%, T3=99.7%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_19-25-19_exp13_task_inference) · [train.log](output\split_mnist\experiments\2026-02-11_19-25-19_exp13_task_inference/train.log) |
| 2026-02-11 | split_mnist\exp12_slice_margin.py | exp12_slice_margin | 26.31% | -91.30% | T0=99.9%, T1=99.0%, T2=95.6%, T3=70.7%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_19-06-57_exp12_slice_margin) · [train.log](output\split_mnist\experiments\2026-02-11_19-06-57_exp12_slice_margin/train.log) |
| 2026-02-11 | split_mnist\exp11_slice_balance.py | exp11_slice_balance | 25.72% | -90.56% | T0=99.3%, T1=92.6%, T2=90.6%, T3=79.8%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_18-48-59_exp11_slice_balance) · [train.log](output\split_mnist\experiments\2026-02-11_18-48-59_exp11_slice_balance/train.log) |
| 2026-02-11 | split_mnist\exp10_adapters.py | exp10_adapters | 20.51% | -98.61% | T0=100.0%, T1=98.6%, T2=99.8%, T3=96.1%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_18-30-47_exp10_adapters) · [train.log](output\split_mnist\experiments\2026-02-11_18-30-47_exp10_adapters/train.log) |
| 2026-02-11 | split_mnist\exp8_attention_plus_si.py | exp8_attention_plus_si | 29.64% | -84.61% | T0=99.9%, T1=97.0%, T2=81.6%, T3=60.0%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-51-42_exp8_attention_plus_si) · [train.log](output\split_mnist\experiments\2026-02-11_17-51-42_exp8_attention_plus_si/train.log) |
| 2026-02-11 | split_mnist\exp7_si.py | exp7_si | 34.15% | -80.88% | T0=95.6%, T1=98.1%, T2=74.3%, T3=55.5%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-33-31_exp7_si) · [train.log](output\split_mnist\experiments\2026-02-11_17-33-31_exp7_si/train.log) |
| 2026-02-11 | split_mnist\exp6_attention_backbone.py | exp6_attention_backbone | 26.27% | -90.57% | T0=100.0%, T1=98.0%, T2=98.8%, T3=65.5%, T4=0.0% | [结果目录](output\split_mnist\experiments\2026-02-11_17-14-37_exp6_attention_backbone) · [train.log](output\split_mnist\experiments\2026-02-11_17-14-37_exp6_attention_backbone/train.log) |

