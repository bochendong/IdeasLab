# ===== Exp13: Task inference 路由 =====
"""轻量 task head：特征 -> 任务概率，推理时用该概率加权各 slice，把 Task-IL 优势迁移到 Class-IL。"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_mnist.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="exp13_task_inference",
        config={
            "lambda_slice": 2.0,
            "lambda_feat": 0.5,
            "lambda_slice_balance": 0.0,
            "lambda_slice_margin": 0.0,
            "use_task_inference": True,
            "lambda_task_inference": 1.0,  # task head CE 损失权重
        },
        script_file=os.path.abspath(__file__),
    )
