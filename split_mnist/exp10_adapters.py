# ===== Exp10: Task-specific Adapters（迁移/适配器）=====
"""每任务一个小型 bottleneck adapter，共享 backbone，组合特征后接 cosine head（无回放）。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp10_adapters",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "freeze_backbone_after_task": -1,
            "ewc_lambda": 0.0,
            "si_lambda": 0.0,
            "use_adapters": True,
        },
        script_file=os.path.abspath(__file__),
    )
