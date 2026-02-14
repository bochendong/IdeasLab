# Split CIFAR-10 实验

与 `split_mnist` 同协议：**5 任务、无真实回放、Class-IL**。  
将 MNIST 上效果好的方法在 CIFAR-10 上复现。

## 设定

- **数据**：CIFAR-10，5 个任务 {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
- **模型**：小 CNN（3 层卷积 + 256 维特征）+ Cosine head
- **训练**：10 epoch/任务，lr=1e-3，batch_size=128
- **评估**：Class-IL 平均准确率、BWT

## 实验列表

| 脚本 | 说明 |
|------|------|
| exp_baseline.py | 无正则 |
| exp_ewc.py | EWC |
| exp_si.py | SI |
| exp_vae_si.py | VAE 伪回放 + SI（对应 MNIST Exp27） |
| exp_dual_discriminator.py | 双判别器（对应 MNIST Exp32） |
| exp_slice_margin.py | 双判别器 + Slice margin（对应 MNIST Exp35） |
| exp_stronger_replay.py | 加强伪回放（对应 MNIST Exp34） |

## 运行

在**项目根目录**执行：

```bash
# 跑全部
python split_cifar10/run_cifar10_experiments.py

# 只跑部分（例如 baseline 和 slice_margin）
python split_cifar10/run_cifar10_experiments.py --only baseline slice_margin

# 列出所有实验
python split_cifar10/run_cifar10_experiments.py --list
```

结果目录：`output/split_cifar10/experiments/`，每个实验子目录含 `config.json`、`metrics.json`、`train.log`。
