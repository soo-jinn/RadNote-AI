$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $env:RADNOTE_ENABLE_OCR) { $env:RADNOTE_ENABLE_OCR = '1' }
if (-not $env:RADNOTE_ALLOW_MODEL_DOWNLOAD) { $env:RADNOTE_ALLOW_MODEL_DOWNLOAD = '0' }
$bind = if ($env:RADNOTE_BIND) { $env:RADNOTE_BIND } else { '127.0.0.1' }
$port = if ($env:RADNOTE_PORT) { $env:RADNOTE_PORT } else { '8000' }
& .venv\Scripts\python.exe -m uvicorn api.main:app --host $bind --port $port
