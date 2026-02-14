#!/usr/bin/env python3
# ===== 只跑 Exp28～Exp32（不跳过，依次跑完）=====
"""用于跑完剩余实验 Exp28、29、30、31、32。每次运行会依次执行这 5 个脚本，各生成新的时间戳目录。"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 只跑这 5 个实验，不检查是否已跑过
REMAINING = [
    ("split_mnist/exp28_proto_adversarial.py", "exp28_proto_adversarial"),
    ("split_mnist/exp29_reverse_distill_si.py", "exp29_reverse_distill_si"),
    ("split_mnist/exp30_slice_gauss_anticollapse_si_proto.py", "exp30_slice_gauss_anticollapse_si_proto"),
    ("split_mnist/exp31_slice_space_adversarial.py", "exp31_slice_space_adversarial"),
    ("split_mnist/exp32_dual_discriminator.py", "exp32_dual_discriminator"),
]


def main():
    for exp_path, run_name in REMAINING:
        path = os.path.join(_PROJECT_ROOT, exp_path)
        if not os.path.isfile(path):
            print(f"[Warn] 未找到: {path}")
            continue
        print(f"\n{'='*60}")
        print(f"Running: {run_name} -> {exp_path}")
        print("="*60)
        ret = subprocess.run([sys.executable, path], cwd=_PROJECT_ROOT)
        print(f"\nCompleted: {run_name} (exit code {ret.returncode})\n")
        if ret.returncode != 0:
            try:
                print(f"[Error] {run_name} 退出码非 0，后续实验仍会继续。")
            except UnicodeEncodeError:
                print(f"[Error] {run_name} exit code != 0, continuing.")
    print("全部完成。")


if __name__ == "__main__":
    main()
