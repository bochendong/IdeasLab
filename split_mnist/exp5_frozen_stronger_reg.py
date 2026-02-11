# ===== Exp5: Frozen + Stronger Reg - 组合策略 =====
"""Frozen Backbone + 更强蒸馏正则"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp5_frozen_stronger_reg",
        config={
            "lambda_slice": 8.0,
            "lambda_feat": 2.0,
            "freeze_backbone_after_task": 0,
            "ewc_lambda": 0.0,
        },
        script_file=os.path.abspath(__file__),
    )
