$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "Python launcher 'py' not found. Install Python 3.12 and ensure it adds the launcher."
}

Write-Host "Creating venv with Python 3.12..."
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
py -3.12 -m venv .venv

Write-Host "Upgrading pip..."
.\.venv\Scripts\python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
.\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"

