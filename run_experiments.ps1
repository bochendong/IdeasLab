# 使用 venv 运行全部实验（PowerShell）
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# 若 venv 在别处，请把下面改为你的 venv 路径，例如: $venv = "C:\path\to\your\venv"
$venv = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venv)) { $venv = Join-Path $ProjectRoot "venv" }

if (Test-Path (Join-Path $venv "Scripts\Activate.ps1")) {
    & (Join-Path $venv "Scripts\Activate.ps1")
    python split_mnist/run_all_experiments.py
} else {
    Write-Host "未找到 venv，请先创建或修改本脚本中的 `$venv 路径。"
    Write-Host "创建示例: python -m venv .venv"
    exit 1
}
