# 在项目根目录执行: .\benchmarks\run_ada_gauss.ps1
# 完整复现（200 epoch/任务，较慢）: .\benchmarks\run_ada_gauss.ps1 -Full
# 后台运行（新开进程，输出到 output 下日志）: .\benchmarks\run_ada_gauss.ps1 -Background
param(
    [switch]$Full,      # 不加则快速试跑 2 epoch/任务
    [switch]$Background # 在独立进程中后台跑，输出写日志
)
$ErrorActionPreference = "Stop"
# 若只传 -Background，在子进程里执行本脚本（不再传 -Background），并退出
if ($Background) {
    $repoRoot = (Split-Path -Parent $PSScriptRoot)
    $logDir = Join-Path $repoRoot "output"
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    $logFile = Join-Path $logDir "adagauss_background.log"
    $errFile = Join-Path $logDir "adagauss_background_err.log"
    $arg = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    if ($Full) { $arg += "-Full" }
    Start-Process powershell -ArgumentList $arg -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $logFile -RedirectStandardError $errFile
    Write-Host "AdaGauss started in background. Log: $logFile"
    exit 0
}
# 与 MNIST 一致：用 Python 脚本跑（脚本内用 sys.executable）
# 优先用项目 .venv 的 Python，这样在 Cursor 前台/无 PATH 时也能跑
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path $repoRoot)) { $repoRoot = (Get-Location).Path }
$venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPy) { $venvPy } else { "python" }
$pyScript = Join-Path $repoRoot "benchmarks\run_ada_gauss.py"
if (-not (Test-Path $pyScript)) {
    Write-Host "Not found: $pyScript"
    exit 1
}
Set-Location $repoRoot
if ($Full) {
    & $pythonCmd $pyScript --full
} else {
    & $pythonCmd $pyScript
}
