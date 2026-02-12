# ===== Exp12: Slice margin =====
"""拉大“正确 slice”与“其它 slice”的 gap（margin loss），使 Class-IL 时更易选对 slice。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp12_slice_margin",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "lambda_slice_balance": 0.0,
            "lambda_slice_margin": 1.0,  # margin 损失权重
            "slice_margin": 0.5,  # 正确 slice max 应比其它 slice max 大至少 0.5
            "use_task_inference": False,
        },
        script_file=os.path.abspath(__file__),
    )
