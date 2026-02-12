# ===== Exp6: Attention Backbone =====
"""Backbone 使用 patch + multi-head self-attention，提升表征与任务区分能力（无回放 Class-IL）。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp6_attention_backbone",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "freeze_backbone_after_task": -1,
            "ewc_lambda": 0.0,
            "si_lambda": 0.0,
            "use_attention_backbone": True,
        },
        script_file=os.path.abspath(__file__),
    )
