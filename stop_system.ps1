# stop_system.ps1 - Windows PowerShell Stop Script
# Stop VitalViewAI services

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║              VitalViewAI System Shutdown                         ║" -ForegroundColor Blue
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Kill processes on port 8000 (API Server)
Write-Host "Stopping API Server (port 8000)..." -ForegroundColor Yellow
$apiProcess = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($apiProcess) {
    Stop-Process -Id $apiProcess -Force -ErrorAction SilentlyContinue
    Write-Host "✅ API Server stopped" -ForegroundColor Green
} else {
    Write-Host "⚠️  API Server not running" -ForegroundColor Yellow
}

# Kill processes on port 8001 (ML Server)
Write-Host "Stopping ML Server (port 8001)..." -ForegroundColor Yellow
$mlProcess = Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($mlProcess) {
    Stop-Process -Id $mlProcess -Force -ErrorAction SilentlyContinue
    Write-Host "✅ ML Server stopped" -ForegroundColor Green
} else {
    Write-Host "⚠️  ML Server not running" -ForegroundColor Yellow
}

# Kill any remaining Python processes running our servers
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*streaming_api_server*" -or 
    $_.CommandLine -like "*ml_server*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                 🛑 SYSTEM STOPPED SUCCESSFULLY! 🛑                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host ""
Write-Host "📝 Logs preserved in logs\" -ForegroundColor Blue

Write-Host ""
Write-Host "🚀 To start again:" -ForegroundColor Blue
Write-Host "   .\start_system.ps1" -ForegroundColor Yellow
Write-Host ""