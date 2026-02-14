# IdeasLab 实验报告

> 实验记录由 ExperimentManager 自动维护

## 环境与运行（Windows）

- **Python** 3.11，venv 后：`pip install -r requirements.txt`
- **PyTorch**：无显卡用 CPU 版（见 `requirements.txt`）；有 NVIDIA 显卡见下「使用 GPU」。运行实验：`python split_mnist/run_all_experiments.py`
- **使用 GPU**：`pip uninstall -y torch torchvision torchaudio` 后，按驱动 CUDA 版本装（例 CUDA 12.1）：`pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`。验证：`python -c "import torch; print(torch.cuda.is_available())"` 为 `True`。详见 `requirements-gpu.txt`。
- **Batch size**：默认 128，占显存约 1G 量级即可跑。若显存有余想加速，可传 `config={"batch_size": 256}` 或运行 CIFAR-10 时加 `--batch_size 256`；若 OOM 可改为 64。

---

## Baseline 超参调参（建议先做）

为公平比较，建议**先调出较好的 baseline**，再**固定 lr、epochs_per_task、batch_size、seed** 跑所有其他方法。

**运行扫描**
- **CIFAR-10**：`python split_cifar10/tune_baseline.py`  
  默认扫 lr∈{1e-3, 3e-3, 5e-3} × epochs∈{8, 10, 12}，共 9 组。可缩小：`--lr 0.001 0.003 --ep 10 12`
- **MNIST**：`python split_mnist/tune_baseline.py`  
  默认扫 lr∈{3e-3, 5e-3, 1e-2} × epochs∈{4, 6, 8}。可指定：`--lr 0.005 --ep 4 6`

脚本结束会打印汇总表（按 Class-IL 排序）并给出**建议的固定配置**（lr、epochs_per_task），把该配置写入 `base_experiment.DEFAULT_CONFIG` 或在各实验中传入 `config={...}` 即可。

**注意事项**
- **Seed**：调参与正式跑都用同一 seed（默认 42），结果才可复现、可比。
- **只看 Class-IL / BWT**：选 Class-IL 尽量高、BWT 不要太差的一组作为“固定配置”；若多组接近，可挑 BWT 略好或训练更稳的。
- **不要为每个方法单独调 lr**：固定 baseline 定下的 lr 和 epochs，其他方法只调**方法专属**超参（如 EWC 的 `ewc_lambda`、SI 的 `si_lambda`）。
- **lambda_slice / lambda_feat**：若 baseline 已含蒸馏，这两项算“共有超参”；若你希望严格一致，调 baseline 时也可顺带扫一两组（如 1.0/2.0），定下后全方法共用。
- **多 seed 报告**：论文若报 mean±std，在定好超参后用 3～5 个 seed 各跑一遍，再算均值与标准差。

---

## 设定与 SOTA 参考

- **设定**：5 任务 Split MNIST，**无回放**，Class-IL 评估。共同：lr=0.005，4 epoch/任务，batch 128，cosine head。
- **文献**：无回放 Class-IL 多在 20%～40%；本仓库目标在不增加回放下逼近该上界并改善 BWT。

---

## 实验结果总览（Task-IL / Class-IL / BWT）

Task-IL：提供任务 ID 时的平均准确率；Class-IL：不提供任务 ID、所有类联合预测。数值见各实验目录 `metrics.json`（Task-IL 由 `task_il_per_task` 汇总）。

| 实验 | Task-IL ↑ | Class-IL ↑ | BWT | 备注 |
|------|------------|------------|-----|------|
| **Exp35** Slice margin | 92.48% | **50.20%** | **-49.92%** | 显式 gap 约束，当前最佳、BWT 最好 |
| **Exp34** 加强伪回放 | 93.41% | **46.77%** | -52.17% | 64 fake/任务、cap 256、vae_epochs 8 |
| **Exp32** 双判别器 | 93.07% | **46.21%** | -57.05% | 特征+slice 双判别器 |
| **Exp27** VAE 伪样本+SI | 97.79% | **45.82%** | -64.59% | Exp9+SI |
| **Exp39** 双判别器+anti-collapse | 93.44% | **45.50%** | -57.43% | AdaGauss 风 |
| **Exp42** 双判别器+slice 一致性 | 92.70% | **45.46%** | -57.01% | TASS 风 |
| **Exp40** 双判别器+EFM | 92.85% | **45.28%** | -60.74% | EFC 风特征漂移正则 |
| **Exp36** 双判别器+原型 | 93.15% | 43.58% | -57.45% | 双重伪回放 |
| **Exp23** 对抗防遗忘 | 97.89% | 43.02% | -65.71% | VAE+判别器 |
| **Exp38** 双判别器+slice 方差 | 89.91% | 41.05% | -60.93% | — |
| **Exp33** 双判别器+SI | 90.00% | 40.83% | -60.86% | — |
| **Exp31** Slice 空间对抗 | 98.50% | 39.04% | -72.91% | — |
| **Exp19** 原型增强+SI | 99.26% | 39.52% | -73.34% | 轻量无 VAE |
| **Exp37** 50% 伪回放 batch | 99.25% | 37.92% | -76.78% | — |
| **Exp41** 双判别器+正交 | 88.42% | 37.30% | -64.68% | PRL 风 |
| **Exp26** Exp23+SI | 94.58% | 37.94% | -68.77% | — |
| **Exp9** VAE 伪样本 | 97.05% | 40.35% | -71.37% | — |
| **Exp21** Dream 回放 | 99.13% | 36.25% | -78.33% | — |
| **Exp25** 高斯+anti-collapse+SI | 99.22% | 35.33% | -78.92% | — |
| **Exp7** SI | 99.19% | 34.15% | -80.88% | — |
| **Exp30** Exp25+原型伪回放 | 99.14% | 33.32% | -80.55% | — |
| **Exp29** 逆蒸馏+SI | 96.97% | 30.14% | -84.71% | — |
| **Exp20** 逆蒸馏 | 98.32% | 29.62% | -86.24% | — |
| Exp2 / Exp4 / Exp8 / Exp15 / Exp24 等 | — | 28%～29% | -84%～-88% | — |
| **Exp28** 原型+对抗（无 VAE） | 99.34% | 27.47% | -89.87% | — |
| Exp14 / Exp17 / Exp22 等 | — | 26%～27% | -89%～-91% | — |
| Baseline | — | 25.69% | -92.21% | 基准 |
| Exp18 / Exp10 / Exp13 | — | 19%～23% | -95%～-98% | 更差或崩塌 |

结果目录：`output/split_mnist/experiments/`、`output/split_cifar10/experiments/` 等（按数据集划分；每实验含 `metrics.json`、`train.log`、**`config.json`**）。后续可扩展 CIFAR-100 等，结构不变。  
**横向对比与超参**：运行 `python scripts/aggregate_results.py` 生成多数据集并排对比与主要超参表，见 [docs/experiment_results.md](docs/experiment_results.md)。

---

## Split CIFAR-10 实验结果（无回放）

- **设定**：5 任务 Split CIFAR-10，**无回放**，Class-IL；小 3 层 CNN + 256 维 cosine head，10 epoch/任务，lr=1e-3，与 MNIST 协议一致。
- **结论**：在 MNIST 上有效的多种方法（双判别器、slice margin、VAE+SI、加强伪回放等）在 CIFAR-10 上**均未超过 baseline**，部分明显更差。可能原因：任务更难、特征中「slice 结构」未必形成、模型容量吃紧、VAE 伪样本质量差、超参沿用 MNIST 未针对 CIFAR-10 调。详见下方「MNIST vs CIFAR-10 简要分析」。

| 实验 | Class-IL ↑ | BWT | 备注 |
|------|------------|-----|------|
| **Baseline** | **27.03%** | -81.89% | 基准（含 slice/feat 蒸馏） |
| **EWC** | 27.03% | -81.89% | 与 baseline 完全一致，疑超参或需重跑 |
| **SI** | 27.03% | -81.89% | 同上 |
| **Slice margin** | 23.32% | -69.84% | BWT 最好，但 Class-IL 低于 baseline |
| **VAE+SI** | 19.01% | -87.14% | 伪 replay 质量差可能拖累 |
| **Stronger replay** | 18.52% | -86.93% | 同上 |
| **Dual discriminator** | 17.01% | -75.15% | 低于 baseline |

结果目录：`output/split_cifar10/experiments/`。运行：`python split_cifar10/run_cifar10_experiments.py`（支持 `--only`、`--list`）。

---

## MNIST vs CIFAR-10 横向对比与超参

以下横向对比表由 `python scripts/aggregate_results.py` 生成（每数据集每方法取最新一次 run）。运行 `python scripts/aggregate_results.py --update-readme` 可刷新下表。

<!-- AGGREGATE_TABLE -->

| 方法 | MNIST Class-IL ↑ | MNIST BWT | CIFAR-10 Class-IL ↑ | CIFAR-10 BWT |
|------|------------------|-----------|----------------------|--------------|
| Baseline | 25.69% | -92.21% | 27.03% | -81.89% |
| EWC | 29.02% | -87.92% | 27.03% | -81.89% |
| SI | 34.15% | -80.88% | 27.03% | -81.89% |
| VAE+SI | 45.82% | -64.59% | 19.01% | -87.14% |
| Dual discriminator | 46.21% | -57.05% | 17.01% | -75.15% |
| Slice margin | 50.20% | -49.92% | 23.32% | -69.84% |
| Stronger replay | 46.77% | -52.17% | 18.52% | -86.93% |

<!-- END_AGGREGATE_TABLE -->

**全部实验及主要超参**见 [docs/experiment_results.md](docs/experiment_results.md)。每实验完整超参见 `output/<数据集>/experiments/<实验目录>/config.json`。


---

**无 replay 下利用 Task-IL 推高 Class-IL（新实验，目标 50%+）**  
核心发现：**我们的方法基本能保持 Task-IL**，即便在 CIFAR-10 上亦然；问题是如何把这一点用到「不提供 task ID」的 Class-IL 设定。新增三类无 replay 实验，均不依赖 VAE/伪样本：

| 实验 key | 做法 | 评估 |
|----------|------|------|
| **task_routing_si** | 训练 task_head(slice_vec)→task_id，推理时先预测任务再在该 slice 内分类（Task-IL 当 Class-IL 用） | 报 Class-IL（标准）与 **Class-IL(routed)** |
| **current_margin_si** | 仅在当前任务真实数据上显式 margin（正确 slice 比其它大 margin）+ SI | 标准 Class-IL |
| **task_routing_margin_si** | 任务路由 + 当前任务 margin + SI 组合 | 报 Class-IL 与 Class-IL(routed) |

运行新实验：`python split_cifar10/run_cifar10_experiments.py --only task_routing_si current_margin_si task_routing_margin_si`。若 **Class-IL(routed)** 明显高于标准 Class-IL，说明「先推断 task 再分类」能有效利用已保持的 Task-IL，可视为无 replay 下将 Task-IL 转化为 Class-IL 的路径。

**MNIST vs CIFAR-10 简要分析**  
MNIST 上 Class-IL 可达 50%+、多种方法优于 baseline；CIFAR-10 上 baseline 约 27%、各方法未超越。主要原因：（1）**slice 假设**在 MNIST 上成立、在 CIFAR-10 上小 CNN 特征更噪，slice 结构未必存在；（2）**模型容量**在 CIFAR-10 上吃紧，正则易削弱可塑性；（3）**遗忘更极端**（CIFAR-10 上几乎只记得最后一任务），纯正则难以挽回；（4）**VAE 伪样本**在 CIFAR-10 上质量差，伪回放易成噪声；（5）**超参**沿用 MNIST，未为 CIFAR-10 单独调。论文中可写：方法在「特征具可分离 slice 结构」的简单设定下有效，在更复杂数据上需更强 backbone / 调参 / 或少量 replay 以验证可迁移性。

---

## 实验总结

- **Split MNIST（无回放）**：当前最佳为 **Exp35（双判别器 + slice margin）**，Class-IL **50.20%**、BWT **-49.92%**。显式约束「正确 slice 比其它大一截」直接对准决策层 gap，收益最大。第二梯队（45%～47%）包括加强伪回放（Exp34）、anti-collapse（Exp39）、slice 一致性（Exp42）、EFM（Exp40）、双判别器+原型（Exp36）等，均在双判别器基线上叠加「更多伪回放 / 防坍缩 / 一致性 / 特征漂移正则」带来增量。双判别器上再叠 SI（Exp33）或固定 50% 伪回放（Exp37）、正交（Exp41）在本设定下未超过基线。
- **Split CIFAR-10（无回放）**：各方法均未超过 baseline（27.03%），可能原因包括任务更难、slice 结构未必形成、VAE 伪样本质量差、超参沿用 MNIST 等；横向对比与超参见 [docs/experiment_results.md](docs/experiment_results.md)。后续可扩展至 CIFAR-100 等数据集，结果目录与汇总脚本均按数据集划分，便于追加。

---

## 方法共性

以上思路在 **Split MNIST** 与 **Split CIFAR-10** 上均做了验证：MNIST 上双判别器 + slice margin 等明显抬升 Class-IL；CIFAR-10 上各方法暂未超过 baseline，可能因任务更难、slice 结构未充分形成或超参沿用 MNIST，可扩展 CIFAR-100 等再验证可迁移性。

- **伪回放**：在新任务阶段仍对「旧类」代理（VAE 伪样本或原型+噪声）做 CE，使旧 slice 持续得到正向更新，缓解被新任务 slice 压过。
- **稳固**：分两层——**参数层**（如 SI 限制重要参数改动）与**表征/决策层**（对抗：特征或 slice 不要一边倒向新任务）。叠加优于单用（如 VAE+SI、特征+slice 双判别）。
- **决策层显式约束**：slice margin 等直接约束「正确 slice 比其它大一截」，与 Class-IL 的失败形式（错误 slice 的 logit 最大）对症。

一句话：**无回放下，既要让旧类仍被看见并得到监督（伪回放），又要限制 backbone/head 被新任务独占（SI 或对抗）；在决策层显式拉大正确 slice 与其它 slice 的 gap 可进一步抬升 Class-IL。**

---

## 总结反思：为什么起效果、解决的是什么问题

**Class-IL 下遗忘的本质**

测试时不提供任务 ID，预测为所有 logit slice 的 argmax。学新任务时：(1) **表征漂移**：backbone 为拟合新数据改变特征分布；(2) **slice 竞争**：当前 batch 多为新任务，旧 slice 很少被正向更新，新 slice 的 logit 易占优；(3) **头被覆盖**：head 朝“放大新 slice”方向更新。结果：旧类样本的「正确 slice」的 max 常小于「最强错误 slice」的 max，被误判，BWT 很负。

**各思路在解决什么**

| 思路 | 解决的问题 | 为何起效 |
|------|------------|----------|
| **伪回放** | 旧 slice 几乎不被更新、被新 slice 压过 | 在新任务阶段仍对旧类（或其代理）做 CE，让旧 slice 持续得到正样本更新，直接缓解 slice 竞争失衡。 |
| **SI** | 重要参数被覆盖、旧解被改写 | 对旧任务重要参数施加惩罚，限制变动，减少对旧 slice/backbone 的覆盖。 |
| **特征空间对抗** | 表征被新任务独占 | 判别器区分新/旧特征，backbone 在旧(伪)样本上骗判别器，使表征不被新任务独占。 |
| **Slice 空间对抗** | 决策层 slice 竞争一边倒 | 判别器看 slice 模式，直接约束「谁占优」，旧 slice 不能彻底被压下去。 |
| **双判别器** | 表征与 slice 均不过度偏向新任务 | 特征判别器 + slice 判别器同时约束，既管表征又管决策，Class-IL/BWT 最佳。 |
| **Slice margin** | 正确 slice 与其它 slice 的 gap 不足 | 显式把「正确 slice 比其它大一截」写进目标，直接对准失败形式。 |

**结论**  
伪回放是起效前提；稳固分参数层（SI）与表征/决策层（对抗），双判别器同时约束两层故更稳；在决策层再加 slice margin 进一步抬升 Class-IL。在更复杂或不同数据集（如 CIFAR-10/100）上，slice 结构未必形成，需更强 backbone 或调参以验证可迁移性。  

---

## 可结合的机器学习方法（拓宽思路）

从更广的机器学习视角，下列方法可与当前「双判别器 + 伪回放 + slice margin」管线结合，用于缓解遗忘、稳定 slice 竞争或提升伪样本质量。按**表征 / 损失与校准 / 采样与数据 / 其他**分类。

| 类别 | 方法 | 与当前管线如何结合 | 预期作用 |
|------|------|---------------------|----------|
| **表征学习** | **对比学习（SupCon / NT-Xent）** | 在 backbone 特征上：同类别拉近、异类推远；旧类用**存储的原型**作 anchor，新类用当前 batch。 | 旧类特征不被新任务挤成一团，与双判别器「混同分布」互补（对比强调类内紧、类间离）。 |
| | **度量学习（Prototypical / Triplet）** | 显式损失：样本到**正确类原型**的距离 < 到最近错误原型的距离 − margin（特征空间）。 | 与 slice margin（logit 空间）双重约束，既管特征又管 logit gap。 |
| | **Barlow Twins / VICReg** | 对（当前 batch 特征 + 伪旧样本特征）做冗余降低 / 去相关。 | 与 anti-collapse 类似，减轻维度坍缩和 task-recency，可叠在 Exp39 上或替代。 |
| | **解耦表示（β-VAE /  disentanglement）** | 在 VAE 上加强 β 或解耦正则，使潜空间更稳定、伪样本更「可控」。 | 伪回放质量提升，旧类 slice 得到更一致的输入分布。 |
| **损失与校准** | **Focal Loss** | 替代或加权 CE（γ>0）：难样本（如 gap 小、易忘的旧类）权重大。 | 自动把优化重点放在「易忘样本」上，与「对 gap 最负任务加 margin 权重」思路一致。 |
| | **Label Smoothing** | CE 的 target 从 one-hot 改为 (1−ε, ε/(K−1), ...)。 | 减轻过拟合、logit 尺度更平滑，可能缓解某一 slice 过度占优。 |
| | **知识蒸馏（软标签）** | 伪旧样本上：除 CE(硬标签) 外，加 KL(当前 logits ∥ 上一任务模型 logits)。 | 旧类 logit 分布与学该任务时一致，避免「赢家 slice」被悄悄换掉（与 TASS 一致性思想接近）。 |
| **采样与数据** | **重要性加权** | 每个旧任务的伪回放损失乘以权重 w_t（如基于上一轮或当前估计的该任务遗忘量）。 | 多采样/多梯度给忘得最狠的任务（如 T0），与「对 T0 多采伪样本」一致。 |
| | **Mixup / CutMix** | 在「真实新任务样本」与「伪旧样本」之间做 mix（同一 batch 内）。 | 新旧边界更平滑、决策更鲁棒，可能让 slice 竞争不那么极端。 |
| | **课程 / 批内比例** | 每个 batch 内新旧比例按「任务序号」或「当前 gap」动态设定（非固定 50%）。 | 与 Exp37 固定 50% 对比；可先多新后多旧，或 gap 负得多的任务多采样。 |
| **不确定性 / 贝叶斯** | **MC Dropout** | 推理或训练时多次前向取平均，得到不确定性；对高不确定性样本加强伪回放或 margin 权重。 | 把「易忘 / 易错」样本识别出来重点约束。 |
| **元学习** | **MAML 风** | 内循环在「旧任务伪样本」上做几步更新，外循环在新任务上更新，使初始化少改即能恢复旧性能。 | 与 SI 不同：从优化轨迹上希望「少步即可恢复旧解」，实现成本较高。 |
| **架构** | **动态容量 / 专家** | 每任务增加少量参数（adapter、专家），冻结或弱更新旧专家。 | 已有 Exp10 adapters；可在双判别器+margin 基线上再试「每任务小头+共享 backbone」。 |

**落地建议**（与现有代码兼容、改动量可控）：  
- **易结合**：**对比学习（原型作 anchor）**、**Focal Loss 替代/加权 CE**、**伪旧样本上的软标签蒸馏**、**Mixup 真实+伪旧**。  
- **中等**：度量学习（特征空间 margin）、Barlow/VICReg 在特征或 slice 向量上、重要性加权 w_t。  
- **需较多改造成本**：MAML、动态专家、MC Dropout 全流程。

**已实现（Exp45–49，基线均为 Exp35）**：  
| 实验 | 方法 | 说明 |
|------|------|------|
| **Exp45** | Focal Loss | CE 改为 Focal（γ=2），难样本权重大 |
| **Exp46** | 软标签蒸馏 | 伪旧样本上 KL(当前 logits ∥ 上一任务模型)，T=2 |
| **Exp47** | Mixup | 真实新 + 伪旧 mix，软标签 CE，α=0.4 |
| **Exp48** | 对比学习 | 伪旧特征与类原型 SupCon，τ=0.1 |
| **Exp49** | 重要性加权 | 伪回放 CE/margin 按任务加权，越早任务权重越大（decay=0.9） |

---

## 从 doc 论文里可借鉴什么（目标 50%+）

`doc/SOTA.md` 与 `doc/README.md` 里列出的顶会无回放 Class-IL 方法（AdaGauss 60%+、PRL 71%、TASS 67%、EFC 58%+ 等）多数在 CIFAR-100/ImageNet 上，和本仓库的 Split MNIST、cosine head、slice 结构并不一一对应，但**思路可直接借鉴**，下面按「易落地 → 需一定改造成本」列可做的新实验。

| 论文 | 核心可借鉴点 | 我们缺什么 / 可做实验 | 建议编号 |
|------|----------------|------------------------|----------|
| **AdaGauss** (NeurIPS’24) | **Anti-collapse**：对特征（或 logit）批协方差做 Cholesky 约束（diag≥1），缓解维度坍缩 → 减轻 task-recency bias（近期任务占满主维度）。 | 我们已有 Exp22/25 在 **slice 空间** 做高斯+anti-collapse；AdaGauss 在**特征空间**做。可在 Exp32 上对 **backbone 特征** 或 **伪旧样本的 slice 向量** 加 \(L_{AC}\)，避免旧类被压成低秩。 | **Exp39**：Exp32 + 特征/slice 批协方差 anti-collapse |
| **EFC** (ICLR’24) | **EFM 正则**：用「重要特征方向」加权约束漂移，\((f_t-f_{t-1})^\top(\lambda E_{t-1}+\eta I)(f_t-f_{t-1})\)，其它方向保持可塑性。 | 我们只有 SI（参数重要性），没有**特征方向**重要性。可对伪旧样本维护上一任务的 \(E\)，在新任务训练时加 EFM 正则，限制重要方向漂移。 | **Exp40**：Exp32 + EFM 特征漂移正则（伪旧样本） |
| **PRL** (NeurIPS’24) | **PGRU**：增量阶段让新类特征与旧类原型在潜空间**正交**，新类填入「预留空间」、少覆盖旧类。 | 我们做过 Exp15 PRL 预留（29%），可能强度不够。可在 Exp32 上显式加：新任务样本的特征与**旧类原型**正交化损失，把新类推离旧类区域。 | **Exp41**：Exp32 + 新类特征与旧原型正交（PGRU 风） |
| **TASS** (CVPR’24) | **显著性/激活一致性**：旧样本上的 attention 或 logit 模式应与学该任务时保存的模板一致，避免「赢家 slice」被换掉。 | 我们已有 slice 判别器（混淆旧/新），没有「旧类上 slice 模式别变」的**一致性**约束。可在伪旧样本上存「正确 slice 的强度/分布」，新任务时加一致性损失。 | **Exp42**：Exp32 + 伪旧样本 slice 模式一致性（TASS 风） |
| **PASS** (CVPR’21) | **原型+高斯增强** 我们已有（Exp19）；其 **SSL（旋转等）** 我们试过 Exp16，收益不明显。 | 可再试：在现有最佳（Exp32）上**只**加轻量 SSL（如旋转预测），不加重架构；或加大原型噪声方差、增加每类采样数。 | 可选：Exp32 + 轻量 SSL / 更强原型增强 |
| **LDC** (ECCV’24) | **可学习漂移补偿**：旧类原型在新特征空间漂了，用小型 projector 映射回「旧空间」再匹配。 | 我们做过 Exp17 LDC（26.89%）。可放在**双判别器之后**：对旧类用 projector 映射后再算 CE/判别，看是否在强基线上有增益。 | 可选：Exp32 + LDC 式 projector 补偿 |

**优先从 doc 落地的 2 个**  
- **Exp39（AdaGauss 风 anti-collapse 在特征/slice 上）**：实现简单，与「task-recency / 维度坍缩」直接对应，且我们已有 slice 版经验。  
- **Exp41（PRL 风新类与旧原型正交）**：实现量适中，与「新类占预留空间、少覆盖旧类」一致，和当前 slice/双判别器互补。

结论：**不是没有可借鉴的**——AdaGauss 的 anti-collapse、EFC 的 EFM、PRL 的正交预留、TASS 的激活一致性，都可以在现有 Exp32/伪回放框架上做**增量式实验**；优先做 Exp39 与 Exp41，再视结果叠 EFM 或 slice 一致性冲 50%+。

## 实验列表

| 时间 | 脚本 | 实验名 | Final Avg Class-IL | BWT | Forgetting | 结果目录 / Log |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-13 | split_cifar10\exp_baseline.py | cifar10_baseline_tune_20260213_1740_lr0.001_ep8 | 27.08% | -81.35% | T0=96.2%, T1=84.2%, T2=86.8%, T3=58.1%, T4=0.0% | [结果目录](output\split_cifar10\experiments\2026-02-13_17-40-47_cifar10_baseline_tune_20260213_1740_lr0.001_ep8) · [train.log](output\split_cifar10\experiments\2026-02-13_17-40-47_cifar10_baseline_tune_20260213_1740_lr0.001_ep8/train.log) |

