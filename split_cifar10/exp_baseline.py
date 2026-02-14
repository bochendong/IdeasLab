# Split CIFAR-10 Baseline（无 EWC/SI/回放）
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from split_cifar10.base_experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        run_name="cifar10_baseline",
        config=None,
        save_model_checkpoint=False,
        script_file=os.path.abspath(__file__),
    )
