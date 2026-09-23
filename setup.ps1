# One-shot environment setup for Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -TorchCuda cu126 -SkipWeights
param(
    [string]$TorchCuda = "cu128",   # cu118 / cu126 / cu128 / cpu - cu128 is required for RTX 50xx (Blackwell)
    [switch]$SkipWeights
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TorchCuda"
& $py -m pip install -r requirements.txt
if ($SkipWeights) { & $py tools/setup_tuber.py } else { & $py tools/setup_tuber.py --weights csn152-ava22 }
& $py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1   then try:  python tests/smoke_test.py"
