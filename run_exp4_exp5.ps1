# 仅运行 Exp4 和 Exp5，输出到 output/*.log
$ProjectRoot = "c:\Users\yumi\Documents\GitHub\IdeasLab"
$Python = "C:\Users\yumi\anaconda3\python.exe"
Set-Location $ProjectRoot
& $Python split_mnist/exp4_ewc.py *> "output/exp4_ewc.log"
& $Python split_mnist/exp5_frozen_stronger_reg.py *> "output/exp5_frozen_stronger_reg.log"
