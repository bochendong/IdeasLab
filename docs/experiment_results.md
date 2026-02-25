# 实验结果汇总（自动生成）

运行 `python scripts/aggregate_results.py` 生成。数据来自 `output/split_mnist/experiments` 与 `output/split_cifar100/experiments`（每实验取最新一次 run）。

## MNIST vs CIFAR-100 横向对比

| 方法 | MNIST Class-IL ↑ | MNIST BWT | CIFAR-100 Class-IL (TAg) ↑ | CIFAR-100 BWT |
|------|------------------|-----------|-----------------------------|---------------|
| Baseline | 25.69% | -92.21% | 27.03% | -81.89% |
| EWC | 29.02% | -87.92% | 27.03% | -81.89% |
| SI | 34.15% | -80.88% | 27.03% | -81.89% |
| VAE+SI | 45.82% | -64.59% | 19.01% | -87.14% |
| Dual discriminator | 46.21% | -57.05% | 17.01% | -75.15% |
| Slice margin | 50.20% | -49.92% | 23.32% | -69.84% |
| Stronger replay | 46.77% | -52.17% | 18.52% | -86.93% |

## 全部实验（含主要超参）

每实验的完整 config 见对应结果目录下的 `config.json`。

| 数据集 | 实验名 | Class-IL ↑ | BWT | 主要超参 |
|--------|--------|------------|-----|----------|
| MNIST | test_split_mnist_dir | 90.00% | -3.00% | lr=0.005 |
| MNIST | exp35_dual_discriminator_slice_margin | 50.20% | -49.92% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp47_slice_margin_mixup | 49.96% | -53.59% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp48_slice_margin_contrastive | 47.76% | -52.66% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp34_dual_discriminator_stronger_replay | 46.77% | -52.17% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=8, vae_n_fake_per_task=64, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp32_dual_discriminator | 46.21% | -57.05% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp27_vae_pseudo_replay_si | 45.82% | -64.59% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp39_dual_discriminator_anticollapse | 45.50% | -57.43% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp42_dual_discriminator_slice_consist | 45.46% | -57.01% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp40_dual_discriminator_efm | 45.28% | -60.74% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp36_dual_discriminator_proto | 43.58% | -57.45% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp23_adversarial_antiforget | 43.02% | -65.71% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp49_slice_margin_importance | 42.51% | -62.55% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp38_dual_discriminator_slice_var | 41.05% | -60.93% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp33_dual_discriminator_si | 40.83% | -60.86% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp45_slice_margin_focal | 40.68% | -60.97% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp9_vae_pseudo_replay | 40.35% | -71.37% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp46_slice_margin_distill | 40.24% | -63.03% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| MNIST | exp19_proto_aug_si | 39.52% | -73.34% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp31_slice_space_adversarial | 39.04% | -72.91% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_slice=0.1, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp26_adversarial_antiforget_si | 37.94% | -68.77% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp37_dual_discriminator_balanced_batch | 37.92% | -76.78% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp41_dual_discriminator_ortho | 37.30% | -64.68% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
| MNIST | exp21_dream_replay | 36.25% | -78.33% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp25_slice_gauss_anticollapse_si | 35.33% | -78.92% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp7_si | 34.15% | -80.88% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp30_slice_gauss_anticollapse_si_proto | 33.32% | -80.55% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp29_reverse_distill_si | 30.14% | -84.71% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp8_attention_plus_si | 29.64% | -84.61% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp20_reverse_distill | 29.62% | -86.24% | lr=0.005, epochs_per_task=4, batch_size=128, vae_epochs=5, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp2_stronger_reg | 29.37% | -86.26% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=8.0, lambda_feat=2.0 |
| MNIST | exp4_ewc | 29.02% | -87.92% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp15_prl_base_reserve | 29.00% | -88.00% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp24_slice_lora | 28.10% | -88.68% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp14_proto_aug | 27.69% | -89.52% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp28_proto_adversarial | 27.47% | -89.87% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp17_ldc_drift | 26.89% | -90.61% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp22_slice_gauss_anticollapse | 26.87% | -90.67% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp12_slice_margin | 26.31% | -91.30% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp6_attention_backbone | 26.27% | -90.57% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp16_pass_ssl | 25.90% | -91.12% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp11_slice_balance | 25.72% | -90.56% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | split_mnist_groupdiff_no_replay_fixed | 25.69% | -92.21% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp18_asymmetric_ce | 23.33% | -95.23% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp10_adapters | 20.51% | -98.61% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp13_task_inference | 19.81% | -97.90% | lr=0.005, epochs_per_task=4, batch_size=128, slice_margin=0.5, lambda_slice=2.0, lambda_feat=0.5 |
| MNIST | exp5_frozen_stronger_reg | 16.65% | -81.14% | lr=0.005, epochs_per_task=4, batch_size=128, lambda_slice=8.0, lambda_feat=2.0 |
| CIFAR-10 | cifar10_baseline | 27.03% | -81.89% | lr=0.001, epochs_per_task=10, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| CIFAR-10 | cifar10_ewc | 27.03% | -81.89% | lr=0.001, epochs_per_task=10, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| CIFAR-10 | cifar10_si | 27.03% | -81.89% | lr=0.001, epochs_per_task=10, batch_size=128, lambda_slice=2.0, lambda_feat=0.5 |
| CIFAR-10 | cifar10_slice_margin | 23.32% | -69.84% | lr=0.001, epochs_per_task=10, batch_size=128, slice_margin=0.3, lambda_margin=0.5, vae_epochs=8, vae_n_fake_per_task=32, lambda_adv_feat=0.1 |
| CIFAR-10 | cifar10_vae_si | 19.01% | -87.14% | lr=0.001, epochs_per_task=10, batch_size=128, vae_epochs=8, vae_n_fake_per_task=32, lambda_slice=2.0, lambda_feat=0.5 |
| CIFAR-10 | cifar10_stronger_replay | 18.52% | -86.93% | lr=0.001, epochs_per_task=10, batch_size=128, vae_epochs=10, vae_n_fake_per_task=64, lambda_slice=2.0, lambda_feat=0.5 |
| CIFAR-10 | cifar10_dual_discriminator | 17.01% | -75.15% | lr=0.001, epochs_per_task=10, batch_size=128, vae_epochs=8, vae_n_fake_per_task=32, lambda_adv_feat=0.1, lambda_adv_slice=0.1, lambda_slice=2.0 |
