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
| **SD-LoRA** | ICLR 2025 | 多 CL benchmark（含大模型/Foundation Model） | Rehearsal-free 下稳定–可塑性兼顾；LoRA 幅度与方向解耦、无需组件选择即可推理。具体百分比起见原文。 | Scalable Decoupled LoRA；无扩展 prompt/LoRA 池、无旧样本；Oral。 | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/92f43b1d33fae4aa1958f75317f0cec1-Abstract-Conference.html) |
| **ACMap** | CVPR 2025 | 五类 benchmark（多数据集） | 与 SOTA 准确率相当，推理时间与最快方法相当；任务适配器合并为单一适配器。具体百分比起见原文。 | Adapter Merging + Centroid Prototype Mapping；exemplar-free、恒定推理时间。 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fukuda_Adapter_Merging_with_Centroid_Prototype_Mapping_for_Scalable_Class-Incremental_Learning_CVPR_2025_paper.html) |
| **CL-LoRA** | CVPR 2025 | 多 benchmark（预训练模型 CIL） | 多数据集上优于 SOTA，更少可训练参数；双适配器（task-shared + task-specific）+ 知识蒸馏与梯度重分配。具体百分比起见原文。 | Continual Low-Rank Adaptation；Rehearsal-free，基于 PTM + PEFT。 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html) |
| **AdaGauss** | NeurIPS 2024 | **CIFAR-100**、**ImageNet 子集**、**CUB-200**、**FGVC-Aircraft** | **CIFAR-100**：Final **60.2%**（ResNet-18）；**ImageNet 子集**：Final **65.0%**（10 任务）。**CUB-200**：ResNet18 66.2%、ConvNext 73.1%、ViT 77.5%（T=20）；**FGVC-Aircraft**：ResNet18 58.5%、ConvNext 62.9%、ViT 60.6%（T=20）。详见原文表。 | 类为潜空间高斯，自适应协方差 + anti-collapse；EFCIL SOTA 级。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html) |
| **DS-AL** | AAAI 2024 | **CIFAR-100**、**ImageNet-100**、**ImageNet-Full** | **CIFAR-100**、**ImageNet-100**、**ImageNet-Full** 上无回放与回放方法可比或更优；5-phase 至 500-phase ImageNet 可扩展。各数据集具体百分比起见原文 Table 与附录。 | 双流解析学习，C-RLS + 补偿流（DAC）。 | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29670) |
| **G-ACIL / GACL** | NeurIPS 2024 | 多数据集（**Generalized CIL**：新旧类混合、未知样本量） | 广义 Class-IL 设定下领先；解析闭式解、权重不变性等价联合训练。具体数据集与准确率见原文。 | 解析学习（无梯度），暴露/未暴露类分解；与 DS-AL 同系。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html) |
| **Prospective Representation Learning (PRL)** | NeurIPS 2024 | 多 benchmark（NECIL 设定） | 优于现有 NECIL SOTA；base 阶段为未来类预留空间，增量阶段对齐潜空间并聚类新类。具体百分比起见原文。 | 前瞻式表示学习，即插即用；缓解无旧样本时的类间冲突。 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html) |
| **LDC** | ECCV 2024 | 多数据集（含半监督 EFCIL） | 监督与半监督无回放 SOTA；可补偿 backbone 更新导致的旧类原型漂移。具体百分比起见原文。 | 可学习漂移补偿（Learnable Drift Compensation）；首篇无回放半监督持续学习。 | [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72667-5_27) |
| **TASS** | CVPR 2024 | **CIFAR-100**、**ImageNet** 等 | 任务自适应显著性监督，缓解跨任务显著性漂移；边界引导显著性 + 任务无关低层信号。具体百分比起见原文。 | Task-Adaptive Saliency Supervision；保持注意力可塑性。 | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Task-Adaptive_Saliency_Guidance_for_Exemplar-free_Class_Incremental_Learning_CVPR_2024_paper.html) |
| **Elastic Feature Consolidation (EFC)** | ICLR 2024 | **CIFAR-100**、**Tiny-ImageNet**、**ImageNet-Subset**、**ImageNet-1K** | **Cold-start** EFCIL 设定下显著优于 SOTA；EFM 正则特征漂移 + 高斯原型减 task-recency 偏差。具体百分比起见原文。 | 经验特征矩阵（EFM）+ 非对称交叉熵；针对首任务数据不足。 | [ICLR 2024](https://openreview.net/forum?id=7D9X2cFnt1) |
| **Split-and-Bridge** | AAAI 2021 | **CIFAR-100**、**CIFAR-10** 等（Class-IL / Split 设定） | 在 **CIFAR-100**、**CIFAR-10** 上优于 KD 类 SOTA；各数据集与任务划分下的具体准确率见原文表与附录。 | 网络部分切分再桥接，缓解 KD 与 CE 竞争。 | [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16991) |
| **PASS** | CVPR 2021 | **CIFAR-100**、**Tiny-ImageNet**、**ImageNet-Subset** | 无 exemplar 下显著优于同期非回放方法，与回放方法可比；原型增强（ProtoAug）+ 自监督。具体百分比起见原文。 | Prototype Augmentation and Self-Supervision；早期代表性 EFCIL。 | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html) |

**说明**：上述准确率均为**无 exemplar / 无回放**设定下的 Class-IL（或原文所称 EFCIL / NECIL）指标；协议（任务数、backbone、epoch）以各论文为准。广义 CIL（G-ACIL）、cold-start（EFC）、半监督（LDC）等为相应设定下的结果。**2025 年**：ICLR 2025 SD-LoRA、CVPR 2025 ACMap / CL-LoRA 为当前检索到的顶会无回放 Class-IL 工作；另 WACV 2025 有 “A Reality Check on Pre-training for Exemplar-Free Class-Incremental Learning” 等 EFCIL 相关论文。

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

---

*文档由检索与仓库 README 整理，具体数字以各论文与本仓库实验为准。*
