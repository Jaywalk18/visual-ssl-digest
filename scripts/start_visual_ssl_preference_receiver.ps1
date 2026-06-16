$ErrorActionPreference = "Stop"

$Port = 8765
$Receiver = "C:\Users\Administrator\Documents\New project\scripts\visual_ssl_preference_receiver.py"
$LogDir = "H:\Desktop\visual_ssl_paper_reports\preferences\logs"

$existing = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stdout = Join-Path $LogDir "receiver.out.log"
$stderr = Join-Path $LogDir "receiver.err.log"

$python = (Get-Command python.exe -ErrorAction Stop).Source
$arguments = @(
    "`"$Receiver`"",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr
