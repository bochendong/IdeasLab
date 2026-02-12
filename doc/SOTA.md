# Split MNIST Class-IL SOTA 与基准（无回放）

本文档汇总 **Class-Incremental Learning（Class-IL）** 在 **Split MNIST**、**无回放（exemplar-free / replay-free）** 设定下的 SOTA 方法与典型数值，供本仓库实验对照。

---

## 1. 设定说明

- **Split MNIST**：MNIST 按类别拆成 5 个顺序任务，每任务 2 类，常见拆法为 [0,1], [2,3], [4,5], [6,7], [8,9]。
- **Class-IL**：测试时**不提供任务 ID**，模型需在**所有已见类别**（10 类）上直接做预测；与 Task-IL（已知任务再在 2 类内选）不同，难度更大。
- **无回放**：不存储旧任务的真实样本（无 exemplar buffer）；允许生成式伪样本（如 VAE 生成）的设定下，部分工作仍称为 “exemplar-free”。

---

## 2. 文献中无回放 Class-IL 的典型范围

- **仅正则/蒸馏（EWC、LwF、SI 等）**：在 5 任务 Split MNIST 上，Final Average Class-IL 多在 **20%～40%** 区间；遗忘与 BWT 普遍较差，旧任务准确率易跌至很低。
- **有回放**：同一 benchmark 下使用 exemplar 或生成式回放，常可达到 **90%+**；本仓库聚焦无回放，不与回放方法直接比数字。

不同论文在**网络结构、epoch、学习率、是否用 VAE/生成模型**上差异较大，故只能给出大致区间，不宜跨论文直接比单点数值。

---

## 3. 顶会中无回放 Class-IL 相关方法（仅列顶会）

仅列 **NeurIPS / ICML / ICLR / CVPR / ECCV / AAAI** 等顶会中与 **exemplar-free / replay-free Class-IL** 直接相关的工作。**数据集与准确率** 以原文/附录为准，下表为检索到的信息摘要。

| 方法 | 会议 | 数据集 | 准确率（Class-IL，无回放） | 要点 | 原文 |
|------|------|--------|----------------------------|------|------|
| **SD-LoRA** | ICLR 2025 | ImageNet-R、ImageNet-A、DomainNet、CIFAR-100、CUB-200 | **ImageNet-R** N=20：Acc 75.26%，AAA 80.22%；**ImageNet-A** N=10：55.96%；**CIFAR-100**：88.01%；**CUB-200**：77.48%。 | LoRA 幅度与方向解耦；无 prompt/LoRA 池、无旧样本；Oral。 | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/92f43b1d33fae4aa1958f75317f0cec1-Abstract-Conference.html) |
| **ACMap** | CVPR 2025 | CIFAR-100、CUB、ImageNet-R、ImageNet-A、VTAB | 与 EASE 相当或略优；40 任务推理约 39× 加速；**CIFAR B0 Inc5**：\(\bar{A}\) 92.01%，\(A_T\) 87.73%；**IN-R B0 Inc5**：\(\bar{A}\) 77.10%，\(A_T\) 70.25%。 | Adapter Merging + Centroid Prototype Mapping；exemplar-free、O(1) 推理。 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fukuda_Adapter_Merging_with_Centroid_Prototype_Mapping_for_Scalable_Class-Incremental_Learning_CVPR_2025_paper.html) |
| **CL-LoRA** | CVPR 2025 | CIFAR-100、ImageNet-R、ImageNet-A、VTAB | **CIFAR-100** T=20：A 91.02%，AT 85.32%；**ImageNet-R** T=40：81.58%；**ImageNet-A** T=5：70.15%；**VTAB** T=10：94.57%。0.3% 可训练参数。 | 双 adapter（task-shared + task-specific）+ KD 与梯度重分配；Rehearsal-free。 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html) |
| **AdaGauss** | NeurIPS 2024 | **CIFAR-100**、**ImageNet 子集**、**CUB-200**、**FGVC-Aircraft** | **CIFAR-100**：Final **60.2%**（ResNet-18）；**ImageNet 子集**：Final **65.0%**（10 任务）。**CUB-200**：ResNet18 66.2%、ConvNext 73.1%、ViT 77.5%（T=20）；**FGVC-Aircraft**：ResNet18 58.5%、ConvNext 62.9%、ViT 60.6%（T=20）。详见原文表。 | 类为潜空间高斯，自适应协方差 + anti-collapse；EFCIL SOTA 级。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html) |
| **DS-AL** | AAAI 2024 | CIFAR-100、ImageNet-100、ImageNet-Full | **CIFAR-100** K=50：\(\bar{A}\) 66.33%，\(A_K\) 58.37%；**ImageNet-Full** K=25：66.81% / 58.10%，优于 RMM；K=500 与 K=5 近乎相同（phase-invariant）。 | C-RLS + DAC 补偿流；5～500 phase 可扩展。 | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29670) |
| **G-ACIL / GACL** | NeurIPS 2024 | CIFAR-100、ImageNet-R、TinyImageNet（Si-Blurry） | **CIFAR-100**：AAUC 57.99%，ALast 70.31%；**ImageNet-R**：AAUC 41.68%；**TinyImageNet**：AAUC 63.14%，ALast 62.68%。Mem=0 优于多数回放。 | 解析学习、闭式解；exposed/unexposed 分解；广义 CIL。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html) |
| **PRL** | NeurIPS 2024 | CIFAR-100、TinyImageNet、ImageNet-Subset、ImageNet-1K | **CIFAR-100** P=5/10/20：71.26% / 70.17% / 68.44%；**TinyImageNet**：58.12% / 57.24% / 54.51%；**ImageNet-1K** P=10：62.74%。即插即用（IL2A +2.18%，PASS +2.75%）。 | Base 阶段为未来类预留空间（PES）；增量阶段 PGRU 对齐潜空间；NECIL。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html) |
| **LDC** | ECCV 2024 | 多数据集（含半监督 EFCIL） | 监督与半监督无回放 SOTA；可补偿 backbone 更新导致的旧类原型漂移。具体百分比起见原文。 | 可学习漂移补偿（Learnable Drift Compensation）；首篇无回放半监督持续学习。 | [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72667-5_27) |
| **TASS** | CVPR 2024 | CIFAR-100、Tiny-ImageNet、ImageNet-Subset | **CIFAR-100** 10 tasks：Avg 67.42%，Last 57.93%；优于 SSRE 约 2.4% / 2.9%；**TASS(PRAKA)** Last 60.04%。即插即用（MUC +20.77% Last）。 | 任务自适应显著性、边界引导 + 任务无关低层；exemplar-free。 | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Task-Adaptive_Saliency_Guidance_for_Exemplar-free_Class_Incremental_Learning_CVPR_2024_paper.html) |
| **EFC** | ICLR 2024 | CIFAR-100、TinyImageNet、ImageNet-Subset、ImageNet-1K | **Cold Start**：CIFAR-100 10 step \(A_{inc}\) 58.58%；ImageNet-Subset 59.94%；优于 FeTrIL、SSRE、PASS 约 8%+。 | EFM 正则 + 高斯原型；Cold-start EFCIL；针对首任务数据不足。 | [ICLR 2024](https://openreview.net/forum?id=7D9X2cFnt1) |
| **Split-and-Bridge** | AAAI 2021 | CIFAR-100、Tiny-ImageNet | **CIFAR-100** 2/5/10/20 tasks：69.6% / 68.62% / 66.97% / 61.12%；**Tiny-ImageNet**：60.52% / 57.16% / 54.81% / 51.63%。优于 STD+WA、Bic、iCaRL。 | **使用 exemplar**；网络切分再桥接，缓解 KD 与 CE 竞争。 | [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16991) |
| **PASS** | CVPR 2021 | CIFAR-100、TinyImageNet、ImageNet-Subset | **CIFAR-100** 10 phases：49.03%（vs MUC 39.80%，+29.3%）；**TinyImageNet** 10 phases：39.28%（vs MUC 26.52%，+25.2%）；与 iCaRL、EEIL、UCIR 相当。 | Non-exemplar：原型 + 高斯 augmentation + SSL；早期 EFCIL 代表。 | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html) |

**说明**：上述准确率均为**无 exemplar / 无回放**设定下的 Class-IL（或原文所称 EFCIL / NECIL）指标；**Split-and-Bridge 使用 exemplar**，列入供对照。协议（任务数、backbone、epoch）以各论文为准。广义 CIL（G-ACIL）、cold-start（EFC）、半监督（LDC）等为相应设定下的结果。**详细摘要见 `doc/README.md`**。**2025 年**：ICLR 2025 SD-LoRA、CVPR 2025 ACMap / CL-LoRA 为当前检索到的顶会无回放 Class-IL 工作；另 WACV 2025 有 “A Reality Check on Pre-training for Exemplar-Free Class-Incremental Learning” 等 EFCIL 相关论文。

---

## 4. 本仓库在当前设定下的位置

- **设定**：5 任务 Split MNIST，无真实样本回放，MLP/轻量 backbone，cosine head，每任务 4 epoch，lr=0.005 等（见 README）。
- **本仓库当前最佳（无回放）**：**Exp9 VAE 伪样本** — Class-IL **40.35%**，BWT -71.37%；其次 Exp7 SI 34.15%、Exp2 更强正则 29.37%、Exp4 EWC 29.02%；Baseline 约 25.69%。
- **对照**：本仓库 40.35% 落在文献中无回放 Class-IL 的常见 **20%～40%** 上界附近；与“仅正则/蒸馏”的典型区间一致，VAE 伪样本在本设定下明显优于仅蒸馏/正则。

---

## 5. 参考与链接（可自行扩展）

- Split MNIST  benchmark 介绍：如 [Emergent Mind - Split-MNIST](https://www.emergentmind.com/topics/split-mnist-task)、Avalanche 文档。
- 无回放 Class-IL（顶会）：**2025** — ICLR SD-LoRA，CVPR ACMap、CL-LoRA；**2024** — NeurIPS AdaGauss / G-ACIL / PRL，AAAI DS-AL，ICLR EFC，CVPR TASS，ECCV LDC；**2021** — AAAI Split-and-Bridge，CVPR PASS；另 WACV 2025 有 EFCIL 预训练相关论文。具体 Split MNIST 数字需查原文或附录。
- 本仓库实验配置与结果：见项目根目录 `README.md` 与 `output/split_mnist/experiments/` 下各实验的 `config.json`、`metrics.json`。
- **各方法详细摘要**：见 `doc/README.md`（含研究背景、核心方法、实验结果、代码链接）。

---

*文档由检索与仓库 README 整理，具体数字以各论文与本仓库实验为准。*
