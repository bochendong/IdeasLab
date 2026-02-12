# ===== Exp11: Slice 平衡 =====
"""约束各任务 slice 的“强度”尽量一致（方差损失），避免新任务 logits 系统性压过旧任务，提升 Class-IL。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp11_slice_balance",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "lambda_slice_balance": 0.5,  # 各 slice mean_max 的方差损失
            "lambda_slice_margin": 0.0,
            "use_task_inference": False,
        },
        script_file=os.path.abspath(__file__),
    )
