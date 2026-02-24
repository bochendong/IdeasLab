# 论文复现：报 ~60% 的无回放 Class-IL 代码

本目录用于克隆并运行**声称 60% 左右**的无回放 Class-Incremental Learning 论文官方代码，便于与仓库内 Split CIFAR-10 结果对比。

## AdaGauss (NeurIPS 2024)

- **论文**：Task-recency bias strikes back: Adapting covariances in Exemplar-Free Class Incremental Learning  
- **设定**：CIFAR-100，10 任务 × 10 类（10x10），无 exemplar，ResNet-32  
- **报告**：60.2% ± 0.9（Class-IL）  
- **代码**：已克隆到 `AdaGauss/`，基于 [FACIL](https://github.com/mmasana/FACIL)

### 环境与运行

**和 MNIST 一样**：在**项目根目录**用你当前 Python（conda/venv 或 `python`）跑：
```bash
# 快速试跑（2 epoch/任务）
python benchmarks/run_ada_gauss.py
# 完整 200 epoch/任务
python benchmarks/run_ada_gauss.py --full
```
PowerShell 也可：`.\benchmarks\run_ada_gauss.ps1`（同上，需终端里能执行 `python`）。

1. **依赖**（与主项目可共用 conda/venv）：
   ```bash
   pip install torch torchvision numpy
   ```
   若需与 README 一致：`torch==2.2.0 torchvision==0.17.0`（可选）

2. **数据**：CIFAR-100 会自动下载到项目根目录的 `data/cifar100`（已改 AdaGauss 内数据路径指向 `../../data`，即 IdeasLab/data）。

3. **跑 CIFAR-100 10×10（完整 200 epoch/任务）**：
   ```bash
   cd benchmarks/AdaGauss
   set PYTHONPATH=src
   python src/main_incremental.py --approach ada_gauss --seed 1 --batch-size 256 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 64 --lamb 10 --use-test-as-val --criterion ce --distillation projected --rotation --normalize --multiplier 32 --distiller mlp --adapter mlp --exp-name 10x10/ --results-path ../../output/adagauss_results
   ```
   Windows PowerShell：
   ```powershell
   cd benchmarks\AdaGauss
   $env:PYTHONPATH = "src"
   python src/main_incremental.py --approach ada_gauss --seed 1 --batch-size 256 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 64 --lamb 10 --use-test-as-val --criterion ce --distillation projected --rotation --normalize --multiplier 32 --distiller mlp --adapter mlp --exp-name 10x10/ --results-path ../../output/adagauss_results
   ```

4. **快速试跑**（每任务 2 epoch，仅验证能跑通）：
   ```bash
   cd benchmarks/AdaGauss
   set PYTHONPATH=src
   python src/main_incremental.py --approach ada_gauss --seed 1 --batch-size 128 --nepochs 2 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --results-path ../../output/adagauss_quick
   ```

### 结果位置

- 完整跑：`output/adagauss_results/`  
- 快速跑：`output/adagauss_quick/`  
- 日志与准确率在对应目录下（FACIL 格式）。

### 与本仓库对比说明

- AdaGauss 是 **CIFAR-100**（10 类/任务），本仓库 Split CIFAR-10 是 **5 任务 × 2 类**，**数据集与任务划分不同**，不能直接比单点数字。  
- 复现 AdaGauss 主要用于确认：在相同设定下能否跑出论文量级（~60%），以作参考。
