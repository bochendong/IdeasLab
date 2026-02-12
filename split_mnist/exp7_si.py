# ===== Exp7: Synaptic Intelligence (SI) =====
"""突触重要性正则：沿训练路径累积 omega，惩罚对旧任务重要的参数变化（无回放 Class-IL）。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp7_si",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "freeze_backbone_after_task": -1,
            "ewc_lambda": 0.0,
            "si_lambda": 1.0,  # SI 强度，可与 EWC 对比调参
        },
        script_file=os.path.abspath(__file__),
    )
