#!/usr/bin/env python3
# ===== 只跑 Exp29～Exp32 =====
"""依次运行 Exp29、30、31、32（不包含 Exp28）。"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REMAINING = [
    ("split_mnist/exp29_reverse_distill_si.py", "exp29_reverse_distill_si"),
    ("split_mnist/exp30_slice_gauss_anticollapse_si_proto.py", "exp30_slice_gauss_anticollapse_si_proto"),
    ("split_mnist/exp31_slice_space_adversarial.py", "exp31_slice_space_adversarial"),
    ("split_mnist/exp32_dual_discriminator.py", "exp32_dual_discriminator"),
]


def main():
    for exp_path, run_name in REMAINING:
        path = os.path.join(_PROJECT_ROOT, exp_path)
        if not os.path.isfile(path):
            print(f"[Warn] Not found: {path}")
            continue
        print(f"\n{'='*60}\nRunning: {run_name}\n{'='*60}")
        ret = subprocess.run([sys.executable, path], cwd=_PROJECT_ROOT)
        print(f"\nDone: {run_name} (exit {ret.returncode})\n")
    print("All done.")


if __name__ == "__main__":
    main()
