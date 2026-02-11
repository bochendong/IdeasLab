# ===== Exp2: Stronger Reg - 更强蒸馏正则 =====
"""LAMBDA_SLICE=8, LAMBDA_FEAT=2，观察仅靠更强正则能否缓解遗忘"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp2_stronger_reg",
        config={
            "lambda_slice": 8.0,
            "lambda_feat": 2.0,
            "freeze_backbone_after_task": -1,
            "ewc_lambda": 0.0,
        },
        script_file=os.path.abspath(__file__),
    )
