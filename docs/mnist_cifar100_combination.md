# MNIST 思路与 CIFAR-100 / AdaGauss 的结合方向

MNIST 上验证有效的做法（slice margin、蒸馏、双判别器、伪回放等）与 AdaGauss 在 CIFAR-100 上的设计（类高斯、adapter、蒸馏）可以多方向结合。下面按「易落地 → 需一定改造成本」列出可做结合与推荐实现顺序。

---

## 一、两边各自的核心

| 来源 | 核心点 |
|------|--------|
| **MNIST（split_mnist）** | 多任务 slice（每任务一个「块」logits）；**slice margin**（正确 slice 的 max 比其它 slice 的 max 大一截）；slice/特征**蒸馏**防漂移；**双判别器**（特征+slice）防新任务独占；VAE/原型**伪回放**；SI/EWC 参数正则。 |
| **AdaGauss（CIFAR-100）** | **类高斯 (μ, Σ)** + 马氏距离分类，旧类不存线性 head；**Adapter** 把旧类分布迁到新 backbone 空间；**projected 蒸馏** 限制特征漂移；anti-collapse 保协方差可逆。 |

---

## 二、可结合方向（按推荐顺序）

### 1. 在 split_cifar100 上直接搬 MNIST 的「决策层约束」

**思路**：CIFAR-100 已是 10 个 head（每任务 10 类），每个 head 等价于 MNIST 的一个「slice」。可以直接加：

- **Slice margin**：正确任务 head 的 max logit ≥ 其它任务 head 的 max + margin，与 MNIST 的 `slice_margin_loss` 同构。
- **Slice 蒸馏**：新任务训练时，旧任务 head 的 logits 尽量接近旧模型（LwF 式），按任务/slice 做，不按类。
- **Slice balance**：各任务 head 的「强度」（如 max logit 的均值）不要方差太大，避免新任务 head 压过旧任务。

**实现**：在 `split_cifar100/base_experiment.py` 里加 `lambda_slice_margin`、`lambda_slice`（蒸馏）、`lambda_slice_balance`，复用 MNIST 的 loss 形式，只把「每 slice 2 类」改成「每 slice 10 类」。无需改 backbone，无需类高斯。

**预期**：Class-IL 比纯 CE 基线提升，与 MNIST 上 slice margin 提升一致；便于和 AdaGauss 的 TAg 曲线对比。

---

### 2. 在 split_cifar100 上做「特征蒸馏」+ 旧模型冻结

**思路**：MNIST 用 `reg_loss_slice`（旧 slice logits + 旧特征）约束新模型。在 CIFAR-100 上：

- 每学完任务 t，存一份 `old_model`（冻结）。
- 训练任务 t+1 时：`loss = CE + λ_feat * MSE(feat, old_model(x))`，可选加 logit 蒸馏（见上）。

与 AdaGauss 的 **projected distillation** 同向（都是限制特征漂移），但我们用「直接 MSE 特征」或单层线性投影即可，不必上 MLP distiller。

**实现**：在 base_experiment 里对 `t>0` 加特征蒸馏项，保留 `prev_model`。可与 slice margin 叠加。

---

### 3. 在 split_cifar100 上做「轻量类原型 + adapter 迁移」

**思路**：借鉴 AdaGauss 的 **adapt_distributions**，但简化：不存协方差，只存**每类均值**（原型）。

- 每学完任务 t：用当前 backbone 在训练集上算每类的特征均值 → 存 `means[0..10*(t+1)-1]`。
- 学任务 t+1 时：backbone 会变，旧类原型会漂移。训练一个 **adapter**（如 Linear(64,64) 或小 MLP）：输入 = 旧 backbone 特征，目标 = 当前 backbone 特征（MSE）。然后用 adapter 迁移旧类原型：`old_means → adapter(old_means)`，得到新空间下的旧类原型。
- 推理：当前 backbone 特征与「当前类原型 + 迁移后的旧类原型」做最近邻（或余弦），即 NCM。

**实现**：在 base_experiment 里维护 `class_means`；每新任务后先训 adapter（只用当前任务数据，目标 = 新特征，输入 = 旧模型特征），再更新旧类 means。分类头可改为「最近原型」或保留线性 head 做混合。无需马氏、无需协方差，实现量适中。

**预期**：旧类表示随 backbone 一起「迁到新空间」，比纯线性 head 更抗遗忘，可逼近 AdaGauss 部分收益。

---

### 4. 双判别器（特征 + 「slice/head」）上 CIFAR-100

**思路**：MNIST 上 **Exp32/35** 用特征判别器 + slice 判别器，让特征和 slice 不要一边倒向新任务。在 CIFAR-100 上：

- **特征判别器**：输入 backbone 特征，预测「来自新任务还是旧任务」（用当前任务数据当新、旧模型/旧数据代理当旧），梯度反转或 GAN 式。
- **Slice/head 判别器**：输入 10 个 head 的 max logit（或 100 维 logits），预测新/旧任务，同样约束「旧 head」仍有效。

**实现**：在 split_cifar100 里加两个小判别器，训练时加对抗损失（lambda_adv_feat、lambda_adv_slice）。数据上若严格无回放，旧任务用「旧模型前向」生成的 logit/特征当伪样本（类似 MNIST 上部分实验）。

**预期**：与 MNIST 一致，缓解新任务独占 representation 和决策层。

---

### 5. 伪回放（原型/VAE）上 CIFAR-100

**思路**：MNIST 上 VAE 伪样本或原型+噪声做 CE 伪回放，让旧 slice 持续有监督。在 CIFAR-100 上：

- **原型伪回放**：每类存均值特征（或原型图），新任务训练时对「旧类原型 + 噪声」前向，对旧 head 做 CE。
- **VAE 伪回放**：用 VAE 生成旧类图像，对新任务训练时混入，对旧 head 做 CE（实现成本高，可后做）。

**实现**：先做原型版——维护每类一个原型特征（或一张代表图），每任务训练时采样噪声加在原型上，过 backbone+head 算旧类 CE。可与 slice margin、蒸馏一起用。

---

### 6. SI / EWC 上 CIFAR-100

**思路**：MNIST 上 SI、EWC 作为参数层正则。在 split_cifar100 上同样对 backbone（和 head）做 EWC 或 SI，限制重要参数改动。

**实现**：从 split_mnist/base_experiment 把 `compute_fisher`、`ewc_penalty`、`si_omega`、`update_si_omega` 等迁到 split_cifar100，在每任务结束后更新 Fisher/omega，下一任务 loss 加正则项。与 slice margin、蒸馏可叠加。

---

## 三、推荐实现顺序（在 split_cifar100 上）

1. **Slice margin + logit 蒸馏**（对应 MNIST Exp35 的决策层部分）：实现快、与 MNIST 直接可比。
2. **特征蒸馏**（旧模型冻结 + MSE(feat, old_feat)）：与 AdaGauss 的蒸馏同向，代码少。
3. **轻量原型 + adapter 迁移**：不实现完整 AdaGauss，只做「均值 + adapter」，验证「旧类可迁移」的收益。
4. **SI 或 EWC**：复用 MNIST 逻辑，看 CIFAR-100 上参数正则的增益。
5. **双判别器 / 伪回放**：在 1～4 稳定后再加，便于消融。

这样 MNIST 上验证的「决策层 margin、蒸馏、对抗、伪回放、参数正则」都能在 CIFAR-100 上有一一对应的实验，并与 AdaGauss 的 TAg 曲线对比，看哪些结合最接近或超过 AdaGauss 的 ~50%～60%。

---

## 四、和 AdaGauss 的对比意义

- **AdaGauss**：用类高斯 + adapter + 蒸馏，**不存旧数据**，旧类靠「分布迁移」维持。
- **我们的结合**：保留**线性 head + 可选原型**，用 **slice margin + 蒸馏 + 可选 adapter 迁原型** 来抗遗忘；若加伪回放则有一点「见旧类」但不需真实回放缓冲。

两者可以并行跑：同一数据与评估（TAw/TAg、acc 矩阵、plot 脚本），对比「纯 AdaGauss」vs「ResNet18 + 多 head + MNIST 式正则（及可选轻量原型+adapter）」的 TAg 与 BWT，便于写进同一张表或同一曲线图。
