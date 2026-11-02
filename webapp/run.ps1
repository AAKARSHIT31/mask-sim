# Start the MaskSim Explorer. Run from the repo root with the venv active.
$ErrorActionPreference = "Stop"
if (-not $env:VIRTUAL_ENV) { Write-Host "Activate your venv first: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow }
pip install -q -r webapp\requirements.txt
Write-Host "`nStarting MaskSim Explorer at http://127.0.0.1:8000 ...`n" -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"
python webapp\server.py
