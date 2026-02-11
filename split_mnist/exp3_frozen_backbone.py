# ===== Exp3: Frozen Backbone - Task 0 后冻结 backbone =====
"""完成 Task 0 后冻结 backbone，仅训练新增 head，保持旧任务特征不变"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp3_frozen_backbone",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "freeze_backbone_after_task": 0,  # Task 0 完成后冻结
            "ewc_lambda": 0.0,
        },
        script_file=os.path.abspath(__file__),
    )
