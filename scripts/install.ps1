# One-command setup for Windows: create the venv, install, verify, configure.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv\Scripts\pip.exe install -e ".[all]"
& .\.venv\Scripts\recon3d.exe setup @args
& .\.venv\Scripts\recon3d.exe doctor

Write-Host ""
Write-Host "Done. Activate the environment with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then try:  python scripts\make_demo_dataset.py --out .\demo\refs --views 9 --masks"
