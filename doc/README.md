# 文献摘要

本目录包含 SOTA.md 中所列无回放 Class-IL 顶会论文的 PDF 及简要总结。

---

## 1. SD-LoRA（ICLR 2025）

**论文**：*SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning*  
**PDF**：`ICLR-2025-sd-lora-scalable-decoupled-low-rank-adaptation-for-class-incremental-learning-Paper-Conference.pdf`

### 研究背景与动机

- **问题**：基于 Foundation Model 的 Class-IL 中，现有 prompt 方法（L2P、DualPrompt、CODA-Prompt 等）和 LoRA 方法（如 InfLoRA）通常需要：
  - 不断扩展 prompt/LoRA 池；
  - 或依赖大量旧任务样本回放；
- **目标**：在**无回放（rehearsal-free）**、**推理高效**、**端到端可优化**的前提下实现可扩展 Class-IL。

### 核心方法

- **核心思想**：将 LoRA 的更新分解为**幅度（magnitude）**和**方向（direction）**，分别学习。
- **公式**：在第 \(t\) 个任务上，输出为  
  \(h' = (W_0 + \alpha_1 A_1 B_1 + \alpha_2 A_2 B_2 + ... + \alpha_t A_t B_t)x\)  
  其中：
  - \(\{\alpha_k\}\) 为可学习的 LoRA 幅度；
  - \(A_k B_k\) 为归一化后的 LoRA 方向；
  - 旧任务的方向 \(\{A_k B_k\}_{k<t}\) 固定，只更新当前任务的 \(A_t B_t\) 和全部 \(\{\alpha_k\}\)。

### 经验发现（为何有效）

1. **低损失区域重叠**：不同任务的最优微调权重在参数空间中彼此接近，因此可以用共享的低秩方向覆盖多个任务。
2. **早期方向更重要**：早期学到的 LoRA 方向扮演主要角色，后期任务的方向更多是微调；\(\alpha_k\) 随 \(k\) 增大而减小。
3. **低损失路径**：SD-LoRA 在参数空间中沿一条低损失路径，最终收敛到所有任务共享的低损失区域，从而缓解灾难性遗忘。

### 两个参数高效变体

- **SD-LoRA-RR**：后期任务使用更低的 LoRA rank，减少参数增长。
- **SD-LoRA-KD**：若新方向可由旧方向线性近似，则用最小二乘将新方向融进已有的幅度，不再新增 LoRA 方向，从而避免参数膨胀。

### 实验结果

- **Benchmark**：ImageNet-R、ImageNet-A、DomainNet、CIFAR-100、CUB-200。
- **Backbone**：ViT-B/16（ImageNet 预训练 / DINO）。
- **主要结果**：
  - ImageNet-R (N=20)：Acc 75.26%，AAA 80.22%，优于 InfLoRA 约 7.68%（Acc）和 4.62%（AAA）；
  - ImageNet-A (N=10)：Acc 55.96%，优于 HiDe-Prompt 约 31.05%；
  - DomainNet (N=5)：Acc 72.82%；
  - CIFAR-100：Acc 88.01%；CUB-200：Acc 77.48%。

### 与 SOTA.md 的对应关系

- **无回放**：✓（无 exemplar buffer）
- **推理高效**：✓（固定单一模型，无需 prompt/LoRA 选择）
- **端到端优化**：✓（全部参数为 CL 目标联合优化）

**代码**：<https://github.com/WuYichen-97/SD-Lora-CL>

---

## 2. ACMap（CVPR 2025）

**论文**：*Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning*  
**PDF**：`Fukuda_Adapter_Merging_with_Centroid_Prototype_Mapping_for_Scalable_Class-Incremental_Learning_CVPR_2025_paper.pdf`

### 研究背景与动机

- **问题**：现有 CIL 方法在**推理时间**与**准确率**之间存在权衡：
  - Prompt 方法（L2P、DualPrompt、CODA-Prompt）需 prompt 选择，推理有额外开销；
  - Adapter 方法如 EASE 为每个任务训练独立 adapter，推理复杂度 O(T)，随任务数增长；
- **目标**：在**无回放（exemplar-free）**前提下，将任务特定 adapter 合并为**单一 adapter**，实现**恒定推理时间**且不牺牲准确率。

### 核心方法

- **Adapter Merging**：将任务特定 adapter 递增合并为共享子空间。
  - 平均合并：\(\bar{\omega}_t = (1 - 1/t)\bar{\omega}_{t-1} + (1/t)\omega_t\)；
  - 共享初始权重：所有任务 adapter 从同一 \(\omega_{init}\) 开始，利于形成低损失区域；
  - 初始权重替换：首个任务训练后，用 \(\omega_1\) 替换 \(\omega_{init}\)，后续任务由此初始化，使训练路径更一致。
- **Centroid Prototype Mapping**：在无旧任务数据时，将旧任务原型对齐到当前 adapter 子空间。
  - 仿射近似：\(P_i(\bar{A}_t) \approx P_i(\bar{A}_i) + \Delta P\)；
  - \(\Delta P = E[P_t(\bar{A}_t) - P_t(\bar{A}_i)]\)（当前任务在相邻子空间中的原型 centroid 差）。
- **Early Stopping**：当任务数超过阈值 L 时停止合并，减少训练成本；L≈10 即可接近无停止时的性能。

### 实验结果

- **Benchmark**：CIFAR-100、CUB、ImageNet-R、ImageNet-A、VTAB。
- **Backbone**：ViT-B/16（ImageNet-21K 预训练）。
- **主要结果**：
  - 与 EASE 准确率相当或略优（除 VTAB 外）；
  - 40 任务下推理时间约为 EASE 的 1/39（约 39 倍加速）；
  - 相较最快基线 SimpleCIL，最终 top-1 提升 16%+，推理时间相当；
  - CIFAR B0 Inc5：\(\bar{A}\) 92.01%，\(A_T\) 87.73%；IN-R B0 Inc5：\(\bar{A}\) 77.10%，\(A_T\) 70.25%。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **推理高效**：✓（单一 adapter，O(1) 推理复杂度）
- **任务适配器合并**：将多任务 adapter 合并为单一 adapter，恒定推理时间

**代码**：<https://github.com/tf63/ACMap>

---

## 3. CL-LoRA（CVPR 2025）

**论文**：*CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning*  
**PDF**：`He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf`

### 研究背景与动机

- **问题**：现有 adapter-based CIL 方法（如 InfLoRA、O-LoRA、EASE 等）为每个任务训练新 adapter，导致：
  - 参数冗余与线性增长；
  - 无法利用任务间的共享知识；
  - 推理时需对每个任务分别前向，计算开销大。
- **目标**：在**无回放（rehearsal-free）**前提下，兼顾**跨任务共享知识**与**任务特定表征**，减少可训练参数与推理计算。

### 核心方法

- **双 Adapter 架构**：
  - **Task-shared adapter**：置于前 \(l\) 个 Transformer block（如 \(l=6\)），学习跨任务共享知识；
  - **Task-specific adapter**：置于后 \(N-l\) 个 block，为每个任务保留独特表征。
- **Task-shared adapter 设计**：
  - 固定随机正交下投影 \(B_s\)（\(B_s B_s^\top = I\)），仅训练上投影 \(A_s\)（零初始化）；
  - Early Exit 知识蒸馏：在 \(l\) 处对 [CLS] 表征做 KD，约束共享 adapter 不遗忘；
  - 梯度重分配：按前一任务 \(A_s\) 各行的 L2 范数重新分配 KD 梯度，保护重要维度。
- **Task-specific adapter 设计**：
  - 可学习 block-wise 缩放因子 \(\{\mu_t^i\}\)：\(z_i = f_\theta^i(z_{i-1}) + \mu_t^i \cdot A_t B_t z_{i-1}\)；
  - 正交约束 \(L_{orth}\)：最小化不同任务 block 权重的重叠，减轻任务间干扰。

### 实验结果

- **Benchmark**：CIFAR-100、ImageNet-R、ImageNet-A、VTAB。
- **Backbone**：ViT-B/16（ImageNet-21K 预训练）。
- **主要结果**（仅 0.3% 可训练参数）：
  - CIFAR-100 (T=20)：A 91.02%，AT 85.32%；
  - ImageNet-R (T=40)：A 81.58%；
  - ImageNet-A (T=5)：A 70.15%；
  - VTAB (T=10)：A 94.57%，优于 EASE 等；
  - 推理复杂度：\(O(l + (N-l)T)\)，优于纯 task-specific 的 \(O(NT)\)。

### 与 SOTA.md 的对应关系

- **无回放**：✓（rehearsal-free）
- **双适配器**：task-shared + task-specific LoRA
- **知识蒸馏与梯度重分配**：保护共享知识
- **参数高效**：0.3% 可训练参数，优于多数 LoRA-based 方法

**代码**：<https://github.com/JiangpengHe/CL-LoRA>

---

## 4. AdaGauss（NeurIPS 2024）

**论文**：*Task-recency bias strikes back: Adapting covariances in Exemplar-Free Class Incremental Learning*  
**PDF**：`NeurIPS-2024-task-recency-bias-strikes-back-adapting-covariances-in-exemplar-free-class-incremental-learning-Paper-Conference.pdf`

### 研究背景与动机

- **问题**：将类表示为潜空间高斯分布的 EFCIL 方法存在：
  - **协方差未适应**：特征提取器更新后，旧类分布发生漂移，已记忆的均值/协方差与真实分布不再匹配；EFC 等仅适应均值，未适应协方差；
  - **Task-recency bias**：维度坍缩（dimensionality collapse）使早期任务协方差秩低于近期任务，矩阵求逆时数值不稳定，导致分类偏向近期任务。
- **目标**：在**无回放（exemplar-free）**前提下，**同时适应均值与协方差**，并缓解 task-recency bias。

### 核心方法

- **高斯分布适应**：每任务训练后，用辅助 adapter 网络 \(\psi_{t-1 \to t}\) 将旧潜空间映射到新潜空间。
  - 对每个旧类 \(c\)，从 \(N(\mu_c, \Sigma_c)\) 采样 \(N\) 个点，经 \(\psi\) 映射后重新计算 \(\mu_{new}, \Sigma_{new}\)，并更新记忆。
- **Projected Distillation**：通过可学习 projector 做特征蒸馏，相比 logit/feature 蒸馏提升表征强度与特征分布。
- **Anti-collapse 损失 \(L_{AC}\)**：对每个 minibatch 协方差的 Cholesky 分解，约束 diag 元素 \(\geq 1\)，使协方差正定，缓解维度坍缩。
- **总损失**：\(L = L_{CE} + L_{AC} + \lambda L_{PKD}\)。

### 实验结果

- **Benchmark**：CIFAR-100、TinyImageNet、ImagenetSubset（从零训练）；CUB200、FGVC-Aircraft（预训练 backbone）。
- **Backbone**：ResNet18。
- **主要结果**：
  - ImagenetSubset (T=10)：\(A_{last}\) 51.1%，\(A_{inc}\) 65.0%，优于 EFC 约 3.7% / 5.1%；
  - ImagenetSubset (T=20)：\(A_{last}\) 42.6%，\(A_{inc}\) 57.4%；
  - CUB200 (T=10)：\(A_{last}\) 55.8%，\(A_{inc}\) 66.2%；
  - FGVC-Aircraft (T=10)：\(A_{last}\) 47.5%，\(A_{inc}\) 58.5%。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **高斯原型**：类为潜空间高斯，均值+协方差均适应
- **Anti-collapse**：缓解 task-recency bias，EFCIL SOTA 级

**代码**：<https://github.com/grypesc/AdaGauss>

---

## 5. DS-AL（AAAI 2024）

**论文**：*DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning*  
**PDF**：`29670-Article Text-33724-1-2-20240324.pdf`

### 研究背景与动机

- **问题**：Analytic Learning (AL) 类 CIL（如 ACIL）用递归线性工具替代迭代优化，在无回放下可达与回放方法相当的性能，但存在：
  - **欠拟合**：仅依赖单一线性投影，拟合能力不足；
  - 大规模数据集（如 ImageNet-Full）上表现更差。
- **目标**：在**无回放（exemplar-free）**前提下，在保持 AL 的闭式解与等价性优势的同时，**提升拟合能力**。

### 核心方法

- **双流架构**：
  - **主流（Main Stream）**：C-RLS（Concatenated Recursive Least Squares），闭式线性解；
  - **补偿流（Compensation Stream）**：DAC（Dual-Activation Compensation）模块，缓解主流的欠拟合。
- **C-RLS**：将 CIL 重写为拼接递归最小二乘，Theorem 1 表明在 backbone 冻结时，**增量训练与联合训练等价**（权重一致）；递归更新仅需上一阶段权重与当前阶段数据，无需历史样本。
- **DAC 模块**：
  - 主流通用 ReLU (\(\sigma_M\))，补偿流用不同激活（如 Tanh \(\sigma_C\)）；
  - 标签为 mainstream 的残差 \(\tilde{Y}_k\)，补偿流拟合主流的 null space；
  - PLC（Previous Label Cleansing）：仅保留当前阶段对应的残差列，避免对旧类施加错误监督。
- **推理**：\(\hat{Y} = X_M \hat{W}_M + C \cdot X_C \hat{W}_C\)，\(C\) 为补偿比例。

### 实验结果

- **Benchmark**：CIFAR-100、ImageNet-100、ImageNet-Full。
- **Backbone**：ResNet-32（CIFAR）、ResNet-18（ImageNet）。
- **主要结果**：
  - CIFAR-100 (K=50)：\(\bar{A}\) 66.33%，\(A_K\) 58.37%；
  - ImageNet-Full (K=25)：\(\bar{A}\) 66.81%，\(A_K\) 58.10%，优于回放方法 RMM（63.93% / 55.50%）；
  - **Phase 不变性**：K=500 时与 K=5 表现几乎相同，体现 C-RLS 的等价性。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **双流解析学习**：C-RLS + 补偿流（DAC）
- **5-phase 至 500-phase 可扩展**：Phase-invariant 性能

**代码**：<https://github.com/ZHUANGHP/Analytic-continual-learning>

---

## 6. GACL（NeurIPS 2024）

**论文**：*GACL: Exemplar-Free Generalized Analytic Continual Learning*  
**PDF**：`NeurIPS-2024-gacl-exemplar-free-generalized-analytic-continual-learning-Paper-Conference.pdf`

### 研究背景与动机

- **问题**：**广义 CIL（GCIL）** 更贴近真实场景：(i) 每任务类别数不固定；(ii) 旧类可在新任务中再次出现；(iii) 各类样本量不平衡。现有 ACL 方法（ACIL、DS-AL 等）假设任务间类别互斥，无法直接用于 GCIL。
- **目标**：在**无回放（exemplar-free）**前提下，将 ACL 的**权重不变性**（incremental 与 joint training 等价）扩展到 GCIL 设定。

### 核心方法

- **Exposed-Unexposed 分解**：任务 \(k\) 中，**exposed** 为已在任务 1..k-1 出现过的类，**unexposed** 为首次出现的类；标签矩阵 \(Y_{train}^k = [\bar{Y}_{train}^k \tilde{Y}_{train}^k]\)。
- **递归闭式解**（Theorem 3.1）：\(\hat{W}_{FCN}^{(k)} = \hat{W}_{unexposed}^{(k)} + \hat{W}_{ECLG}^{(k)}\)
  - **\(\hat{W}_{unexposed}\)**：仅由 unexposed 类数据递归更新；
  - **\(\hat{W}_{ECLG}\)**（Exposed Class Label Gain）：\(\hat{W}_{ECLG} = [R_k X^{(B)\top}_k \bar{Y}_{train}^k \ 0]\)，捕获 exposed 类带来的增益；若本任务无 exposed 类则为 0。
- **ACIL 为特例**：当 \(\forall k, \bar{Y}_{train}^k \in \mathbb{R}^{* \times 0}\) 时，GACL 退化为 ACIL。
- **无回放**：仅保存自相关记忆矩阵 \(R_k\)，无需存储样本。

### 实验结果

- **Benchmark**：CIFAR-100、ImageNet-R、Tiny-ImageNet，**Si-Blurry** 设定（rD=50% 等）。
- **Backbone**：DeiT-S/16（冻结）。
- **主要结果**（Mem=0，EFCIL）：
  - CIFAR-100：AAUC 57.99%，AAvg 56.24%，ALast 70.31%；
  - ImageNet-R：AAUC 41.68%，AAvg 47.30%，ALast 42.22%；
  - Tiny-ImageNet：AAUC 63.14%，AAvg 69.32%，ALast 62.68%；
  - 优于 SLDA、MVP、LwF 等 EFCIL，及多数回放方法（Mem=500 时）。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **解析学习**：闭式解、无梯度、权重不变性等价联合训练
- **广义 CIL**：暴露/未暴露类分解，支持新旧类混合、未知样本量

**代码**：<https://github.com/CHEN-YIZHU/GACL>

---

## 7. PRL（NeurIPS 2024）

**论文**：*Prospective Representation Learning for Non-Exemplar Class-Incremental Learning*  
**PDF**：`NeurIPS-2024-prospective-representation-learning-for-non-exemplar-class-incremental-learning-Paper-Conference.pdf`

### 研究背景与动机

- **问题**：现有 NECIL 方法多在**新任务到来后**才处理新旧类冲突，缺少旧数据时难以平衡。Base 阶段传统训练会让当前类占满嵌入空间，增量阶段新类涌入导致与旧类重叠。
- **目标**：在**无回放（non-exemplar）**前提下，采用**前瞻式**表征学习：Base 阶段预先为未来类预留空间，增量阶段将新类嵌入到预留空间，减小对旧类的冲击。

### 核心方法

- **Preemptive Embedding Squeezing (PES)**（Base 阶段）：
  - 类内聚拢 + 类间预留分离；
  - \(L_{PES} = (1-s) + \lambda(1+d)\)：\(s\) 为类内余弦相似度（越大越好），\(d\) 为类间余弦相似度（越小越好）；
  - 当前类分布被压缩，为未来类预留空间。
- **Prototype-Guided Representation Update (PGRU)**（增量阶段）：
  - 用保存的旧类原型引导新类嵌入；
  - 将新类特征与旧类原型投影到 latent space，最小化 \(L_{ort}\)：新类特征与旧原型的正交性；
  - \(L_{align}\)：用 MSE 将当前 embedding 空间与 latent space 对齐，使表征更新受 latent space 约束；
  - 新类尽量落在预留空间，减少对旧类的干扰。
- **即插即用**：可叠加到 PASS、IL2A、PRAKA 等 NECIL baseline。

### 实验结果

- **Benchmark**：CIFAR-100、TinyImageNet、ImageNet-Subset、ImageNet-1K。
- **Backbone**：ResNet-18。
- **主要结果**：
  - CIFAR-100 (P=5/10/20)：71.26% / 70.17% / 68.44%；
  - TinyImageNet (P=5/10/20)：58.12% / 57.24% / 54.51%；
  - ImageNet-Subset (P=5/10/20)：72.85% / 71.54% / 66.88%；
  - ImageNet-1K (P=10)：62.74%；
  - 叠加 PRL：IL2A +2.18%（P=5），PASS +2.75%（P=5）。

### 与 SOTA.md 的对应关系

- **无回放**：✓（non-exemplar / NECIL）
- **前瞻式表示学习**：Base 阶段为未来类预留空间，增量阶段对齐潜空间并聚类新类
- **缓解无旧样本时的类间冲突**

**代码**：<https://github.com/ShiWuxuan/NeurIPS2024-PRL>

---

## 8. TASS（CVPR 2024）

**论文**：*Task-Adaptive Saliency Guidance for Exemplar-free Class Incremental Learning*  
**PDF**：`Liu_Task-Adaptive_Saliency_Guidance_for_Exemplar-free_Class_Incremental_Learning_CVPR_2024_paper.pdf`

> **说明**：LDC（ECCV 2024）的 PDF 不在 doc 目录中，此处跳过，直接总结 TASS。

### 研究背景与动机

- **问题**：EFCIL 中模型注意力会随新任务**漂移（saliency drift）**到新类相关特征，导致对旧类显著区域的关注下降，引发灾难性遗忘。单纯对注意力做蒸馏存在语义鸿沟，且缺乏可塑性。
- **目标**：在**无回放（exemplar-free）**前提下，通过**任务自适应显著性监督**缓解显著性漂移，兼顾注意力可塑性与稳定性。

### 核心方法

- **Boundary-guided 中层次显著性**：用 CSNet 等生成边界图，经膨胀得到 \(B_d(x)\)，在 backbone 三个阶段用 Grad-CAM 生成学生显著性图，并在膨胀边界区域施加 BCE 损失 \(L_{dbs}\)，避免显著性漂入背景。
- **任务无关低层辅助监督**：在 backbone 后接 decoder \(D_\psi\)，预测 saliency 与 boundary；用 \(L_{lms} = \|D_\psi(F_\theta(x)) - A(x)\|_2/\sqrt{N}\) 做低层蒸馏，提供跨任务稳定的任务无关先验。
- **Saliency 噪声注入与恢复**：在部分特征通道注入随机椭圆噪声，模拟未来任务的显著性漂移，训练模型去噪，增强对真实漂移的鲁棒性。
- **总损失**：\(L_{all} = L_{CIL} + L_{lms} + L_{dbs}\)；\(L_{CIL}\) 内可包含 saliency noise injection。

### 实验结果

- **Benchmark**：CIFAR-100、Tiny-ImageNet、ImageNet-Subset。
- **Backbone**：ResNet-18。
- **主要结果**：
  - CIFAR-100 (10 tasks)：Avg 67.42%，Last 57.93%，优于 SSRE 约 2.4% / 2.9%；
  - ImageNet-Subset (10 tasks)：Avg 72.60%，Last 57.93%；
  - **即插即用**：+TASS 对 MUC、IL2A、PASS、SSRE 均有明显提升（如 MUC +20.77% Last）；
  - TASS(PRAKA)：CIFAR-100 10-task Last 60.04%。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **任务自适应显著性监督**：缓解跨任务显著性漂移
- **边界引导 + 任务无关低层信号**：保持注意力可塑性

**代码**：<https://github.com/scok30/tass>

---

## 9. EFC（ICLR 2024）

**论文**：*Elastic Feature Consolidation for Cold Start Exemplar-Free Incremental Learning*  
**PDF**：`1311_Elastic_Feature_Consolida.pdf`

### 研究背景与动机

- **问题**：**Cold Start** EFCIL 中首任务数据不足，需 backbone 保持可塑性以适配新任务；但 feature distillation 对各方向均匀约束，牺牲可塑性；冻结 backbone 的方法（如 FeTrIL）在 Cold Start 下表现差。
- **目标**：在**无回放（exemplar-free）**前提下，对**重要方向**约束特征漂移，对**其他方向**保留可塑性，并缓解 task-recency bias。

### 核心方法

- **Empirical Feature Matrix (EFM)**：在特征空间构建类似 Fisher 的二阶近似，\(E_t = \mathbb{E}_{x \sim X_t}[E_{f_t(x)}]\)，其中 \(E_{f_t(x)}\) 与 \(\partial \log p / \partial f\) 相关；EFM 诱导伪度量，表示对预测影响大的特征方向。
- **EFM 正则**：\(L_{EFM} = \mathbb{E}[(f_t(x) - f_{t-1}(x))^\top (\lambda_{EFM} E_{t-1} + \eta I)(f_t(x) - f_{t-1}(x))]\)；仅在重要方向抑制漂移，其他方向允许更多漂移。
- **Asymmetric Prototype Replay (PR-ACE)**：\(L_{PR-ACE} = L_{ce}(x_t, y_t)|_{C_t} + L_{ce}((\tilde{p}, y_{\tilde{p}}) \cup (\hat{x}_t, \hat{y}_t))|_{C_{1:t}}\)；当前任务数据仅训当前类头，原型与当前数据混合后训全部类头，平衡新旧类。
- **原型漂移补偿**：用 EFM 加权当前任务特征漂移更新原型，\(w_i \propto \exp(-(f_i - p_c)^\top E_{t-1}(f_i - p_c)/(2\sigma^2))\)。
- **高斯原型**：从 \(N(p_c, \Sigma_c)\) 采样增强原型。

### 实验结果

- **Benchmark**：CIFAR-100、Tiny-ImageNet、ImageNet-Subset、ImageNet-1K；**Warm Start** 与 **Cold Start** 两种设定。
- **Backbone**：ResNet-18。
- **主要结果**（Cold Start）：
  - CIFAR-100 (10 step)：\(A_{step}\) 43.62%，\(A_{inc}\) 58.58%；
  - ImageNet-Subset (10 step)：\(A_{step}\) 47.38%，\(A_{inc}\) 59.94%；
  - Cold Start 下显著优于 FeTrIL、SSRE、PASS（约 8%+ \(A_{inc}\)）。

### 与 SOTA.md 的对应关系

- **无回放**：✓（exemplar-free）
- **经验特征矩阵（EFM）**：正则特征漂移
- **高斯原型**：减 task-recency 偏差
- **Cold-start**：针对首任务数据不足

**代码**：<https://github.com/simomagi/elastic_feature_consolidation>

---

## 10. Split-and-Bridge（AAAI 2021）

**论文**：*Split-and-Bridge: Adaptable Class Incremental Learning within a Single Neural Network*  
**PDF**：`16991-Article Text-20485-1-2-20210518.pdf`

> **说明**：本方法使用 **exemplar**（回放样本），非 exemplar-free；可与 WA 等推理策略结合。

### 研究背景与动机

- **问题**：CIL 需学习三类知识——intra-old、intra-new、cross-task。标准 KD 方法中，KD 损失常比 CE 损失更主导优化，导致 intra-new 与 cross-task 难以充分学习。
- **目标**：在**单网络**内通过**部分切分**与**桥接**，缓解 KD 与 CE 的竞争，使三类知识更均衡地学习。

### 核心方法

- **两阶段**：Split 阶段 + Bridge 阶段。
- **Split 阶段**：
  - 将网络上层部分拆成两个分支 \(\theta_o\)（旧任务）与 \(\theta_n\)（新任务），共享底层 \(\theta_s\)；
  - 旧分支用 \(L_{kd}\) 蒸馏，新分支用 Localized CE \(L_{lce}\)（仅对新类 logit 做 softmax）分别学习，减少 KD 与 CE 的干扰；
  - **权重稀疏化**：通过 \(L_{kd} + L_{lce} + \gamma \sum (||W_{o,n}||^2 + ||W_{n,o}||^2)\) 使跨分区权重趋于零，再显式断开。
- **Bridge 阶段**：将断开权重零初始化后重新连接，用标准 \(\lambda L_{kd} + (1-\lambda)L_{ce}\) 学习 cross-task 知识；此时 KD 参考模型为 \(\langle\theta_s, \theta_o\rangle_t\)。
- **Exemplar**：使用 \(M_t\) 存储旧任务样本，与 \(D_t\) 一起参与训练。

### 实验结果

- **Benchmark**：CIFAR-100、Tiny-ImageNet。
- **Backbone**：ResNet-18。
- **主要结果**（S&B + WA）：
  - CIFAR-100 (2/5/10/20 tasks)：69.6% / 68.62% / 66.97% / 61.12%（平均准确率）；
  - Tiny-ImageNet (2/5/10/20 tasks)：60.52% / 57.16% / 54.81% / 51.63%；
  - 优于 STD+WA、STD+Bic、STD+iCaRL、DD+WA 等 KD 类方法。

### 与 SOTA.md 的对应关系

- **非 exemplar-free**：使用 exemplar 回放
- **网络部分切分再桥接**：缓解 KD 与 CE 竞争
- **CIFAR-100、CIFAR-10 上优于 KD 类 SOTA**

**代码**：<https://github.com/bigdata-inha/Split-and-Bridge>

---

## 11. PASS（CVPR 2021）

**论文**：*Prototype Augmentation and Self-Supervision for Incremental Learning*  
**PDF**：`Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf`

> **Non-exemplar**：不存储原始样本，仅保存每类一个原型（特征均值），内存占用极低。

### 研究背景与动机

- **问题**：CIL 面临 (1) 灾难性遗忘；(2) **任务级过拟合**——特征偏于当前任务，对后续任务泛化差。
- **目标**：在**不存 exemplar、不用生成模型**的前提下，维持决策边界并学习更可迁移的特征。

### 核心方法

- **Prototype Augmentation (protoAug)**：
  - 为每个旧类保存一个**原型** \(\mu_{t,k}\)（该类的特征均值）；
  - 学习新任务时，用 \(\tilde{F}_{t_{old},k} = \mu_{t_{old},k} + e \cdot r\)（\(e \sim \mathcal{N}(0,1)\)）对旧类原型做高斯扰动；
  - 将 augmented 原型与新类特征一起送入统一分类器，缓解决策边界偏移与类别不平衡。
- **Self-Supervised Learning (SSL)**：
  - 采用旋转标签增强（90°/180°/270°）将 K 类扩展为 4K 类，学习更 task-agnostic 的特征；
  - 降低任务级过拟合，使不同任务在参数空间中更接近。
- **KD**：对特征提取器做 \(L_{kd} = \|F_t(X'_t;\theta_t) - F_{t-1}(X'_t;\theta_{t-1})\|\)  regularization，缓解原型与更新后特征提取器的 mismatch。

### 实验结果

- **Benchmark**：CIFAR-100、TinyImageNet、ImageNet-Subset。
- **Backbone**：ResNet-18。
- **主要结果**（vs 非 exemplar 方法）：
  - CIFAR-100 10 phases：PASS 49.03%，MUC 39.80%，提升约 29.3%；
  - TinyImageNet 10 phases：PASS 39.28%，MUC 26.52%，提升约 25.2%；
  - 与 exemplar-based 方法（iCaRL、EEIL、UCIR）结果相当。

### 与 SOTA.md 的对应关系

- **Non-exemplar**：原型 + 高斯 augmentation，无原始样本
- **Prototype Augmentation + Self-Supervision**
- **显著优于非 exemplar 方法，与 exemplar 方法相当**

**代码**：<https://github.com/Impression2805/CVPR21_PASS>
