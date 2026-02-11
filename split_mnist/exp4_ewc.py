# ===== Exp4: EWC - Elastic Weight Consolidation =====
"""每个任务结束后在自身数据上计算 Fisher，训练新任务时对重要参数施加二次惩罚"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp4_ewc",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "freeze_backbone_after_task": -1,
            "ewc_lambda": 5000.0,  # EWC 强度，可调
        },
        script_file=os.path.abspath(__file__),
    )
