# AdaGauss 结果说明

跑完 AdaGauss（如 `python benchmarks/run_ada_gauss.py --full`）后，结果在 `output/adagauss_results/` 下。这里说明每个结果文件的含义。

---

## 结果目录结构

一次完整实验会生成类似：

```
output/adagauss_results/
└── cifar100_icarl_ada_gauss_10x10/
    ├── args-<timestamp>.txt          # 本次实验的全部超参（JSON 一行）
    ├── stdout-<timestamp>.txt        # 标准输出（训练/测试日志）
    ├── stderr-<timestamp>.txt        # 标准错误
    ├── raw_log-<timestamp>.txt       # 原始日志
    └── results/
        ├── acc_taw-<timestamp>.txt   # TAw 准确率矩阵
        ├── acc_tag-<timestamp>.txt   # TAg 准确率矩阵
        ├── avg_accs_taw-<timestamp>.txt   # 各步的 TAw 平均准确率（一行）
        ├── avg_accs_tag-<timestamp>.txt   # 各步的 TAg 平均准确率（一行）
        ├── wavg_accs_taw-<timestamp>.txt  # 各步的 TAw 加权平均准确率（一行）
        └── wavg_accs_tag-<timestamp>.txt  # 各步的 TAg 加权平均准确率（一行）
```

---

## 核心概念：TAw 和 TAg

- **TAw（Task-Aware）**  
  评估时**已知样本属于哪个任务**，只用该任务对应的分类头/专家做预测。  
  相当于“理想多任务头”的上界，**不是**真实 Class-IL 设定。

- **TAg（Task-Agnostic）**  
  评估时**不告知任务**，在所有已学类别中选预测得分最高的类。  
  这才是**真正的 Class-IL 设定**，论文里主报的指标（如 CIFAR-100 10×10 约 60.2%）一般是 **TAg**。

---

## 各结果文件含义

### 1. `acc_taw` / `acc_tag`（10×10 矩阵）

- **行 `t`**：当前已经学完到第 `t` 个任务（0~9）。
- **列 `u`**：在**任务 `u`** 的测试集上评估准确率。
- **有效区域**：下三角。`acc[t, u]` 只在 `u <= t` 时有意义（任务 `u` 已学过）；其余为 0。

**读法示例**（10×10）：

- **对角线** `acc[t, t]`：学完任务 `t` 后，在**当前任务 `t`** 上的准确率（当前任务表现）。
- **非对角线** `acc[t, u]`（u < t）：学完任务 `t` 后，在**旧任务 `u`** 上的准确率（体现遗忘）。

你这次跑完的最后一行（学完 10 个任务后）大致为：

- **acc_taw** 最后一行：约 `0.946, 0.740, 0.877, …, 0.821` → 各任务上的 TAw 准确率。
- **acc_tag** 最后一行：约 `0.574, 0.371, 0.506, …, 0.571` → 各任务上的 TAg 准确率（Class-IL）。

---

### 2. `avg_accs_taw` / `avg_accs_tag`（一行，10 个数）

- 每个位置 `t`（第 1~10 列）对应：**学完任务 `t` 后**，在**所有已学任务 0, 1, …, t** 上的准确率**简单平均**。
- 公式：`avg_accs[t] = (acc[t,0] + acc[t,1] + … + acc[t,t]) / (t+1)`。

你这次结果示例：

- **avg_accs_taw**：约 `0.952, 0.831, 0.845, …, 0.841` → 平均 TAw 约 84.1%。
- **avg_accs_tag**：约 `0.952, 0.746, 0.687, …, 0.495` → 平均 TAg 约 **49.54%**（即常说的“平均 Class-IL 准确率”）。

---

### 3. `wavg_accs_taw` / `wavg_accs_tag`（一行，10 个数）

- 与 `avg_accs_*` 类似，但按**每个任务的类别数**加权平均。
- 在 CIFAR-100 10×10 中，每个任务都是 10 类，权重相同，所以 **wavg 和 avg 数值会一样**；若每任务类别数不同，wavg 才不同。

---

## 日志里的 TAw acc / TAg acc / forg

训练和测试时打印的格式类似：

```text
>>> Test on task  u : loss=... | TAw acc= xx.x%, forg= yy.y%| TAg acc= zz.z%, forg= ww.w% <<<
```

- **TAw acc**：当前步在任务 `u` 上的 TAw 准确率（对应矩阵里的一个元素）。
- **TAg acc**：当前步在任务 `u` 上的 TAg 准确率。
- **forg（Forgetting）**：在任务 `u` 上的**遗忘量**  
  `forg = 历史上在任务 u 的最高准确率 − 当前准确率`  
  越大表示该任务忘得越多。

---

## 总结：你最该看哪几个数？

| 关心的问题           | 看哪个结果 |
|----------------------|------------|
| Class-IL 最终表现    | **acc_tag** 最后一行，或 **avg_accs_tag** 最后一个数（如 49.54%） |
| 各任务分别的 Class-IL | **acc_tag** 最后一行的 10 个数 |
| 多任务上界（TAw）    | **acc_taw** 最后一行 或 **avg_accs_taw** 最后一个数 |
| 遗忘程度             | 日志里的 **forg**，或比较 **acc_tag** 里同一列在不同行的下降 |

论文中 CIFAR-100 10×10 报的约 **60.2%** 是 **TAg**（Class-IL）指标；你本地若略低可能与 seed、epoch、数据增强等设置有关。

---

## 为什么 AdaGauss 能在无回放下维持约 50%～60% Class-IL？

和「多任务线性 head + CE」的基线相比，AdaGauss 的核心设计让**旧类的分类能力不依赖可被新任务覆盖的权重**，而是依赖**可随 backbone 漂移而一起更新的类级表示**。主要有以下几点。

### 1. 分类器不是线性 head，而是「类高斯 + 马氏距离」

- 每个类存的是 **均值 μ 和协方差 Σ**（在 64 维特征空间），推理时用 **马氏距离**（或等价的高斯 log 概率）在 100 个类上做最近类预测。
- 旧类的「分类器」就是这些 (μ, Σ)，**不存在 backbone 后面的线性层里**，所以新任务加新 head、只对新类做 CE 时，**不会直接覆盖旧类的决策边界**。

### 2. 原型/分布适应（adapt_distributions）

- Backbone 每学完一个新任务都会更新，旧类特征分布会**漂移**；若仍用旧 backbone 时算出的 (μ, Σ)，在新 backbone 下就会错位。
- AdaGauss 在每新任务后训练一个 **adapter**：把**旧 backbone 的特征**映射到**当前 backbone 的特征**（MSE 拟合），然后用 adapter 把旧类的高斯「搬」到新空间：从旧 (μ_old, Σ_old) 采样 → adapter → 在新空间重新估计 (μ_new, Σ_new)。
- 这样**不需要旧数据**，旧类原型也能跟着 backbone 的漂移一起更新，Class-IL 时旧类仍有一致的表示。

### 3. 蒸馏（projected distillation）

- 训练新任务时，当前特征要拟合**旧模型**的特征（或经 distiller 的投影），即 `loss + λ * MSE(distiller(new_feat), old_feat)`。
- 约束新 backbone 不要偏离旧 backbone 太远，**减轻特征空间的灾难性漂移**，使 adapter 更容易把旧分布迁到新空间，且旧类马氏距离仍可用。

### 4. Anti-collapse 与协方差收缩

- 用 anti-collapse 等损失避免特征坍缩到奇异，并做协方差的 shrink，保证 **Σ 可逆、马氏距离稳定**，避免数值或几何上失效。

### 5. 和「多 head + CE」基线的对比

- 基线：旧类信息只存在**共享 backbone + 旧任务的线性 head** 里；新任务训练会更新 backbone 和（若共享）head，旧类决策边界容易被破坏，Class-IL 容易掉到很低。
- AdaGauss：旧类信息存在**类级 (μ, Σ)** 里，且通过 **adapter 迁到新空间**、通过**蒸馏限制漂移**，所以无回放也能在 CIFAR-100 10×10 上维持约 50%～60% 的 TAg。
