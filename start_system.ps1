# start_system.ps1 - Windows PowerShell Start Script
# VitalViewAI System Startup for Windows

Write-Host "================================================================" -ForegroundColor Blue
Write-Host "              VitalViewAI System Startup                        " -ForegroundColor Blue
Write-Host "================================================================" -ForegroundColor Blue
Write-Host ""

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if data exists
if (-not (Test-Path "data\processed\features_multi.csv")) {
    Write-Host "Training data not found. Generating..." -ForegroundColor Yellow
    python generate_diverse_training_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to generate data" -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: Training data generated" -ForegroundColor Green
}

# Check if model exists
if (-not (Test-Path "models\xgboost_model.pkl")) {
    Write-Host "Model not found. Training..." -ForegroundColor Yellow
    python train_models_improved.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to train model" -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: Model trained" -ForegroundColor Green
}

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Blue
Write-Host ""

# Start API server in new window
Write-Host "Starting Streaming API Server (port 8000)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python streaming_api_server.py"
Start-Sleep -Seconds 3

# Start ML server in new window
Write-Host "Starting ML Server (port 8001)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python ml_server.py"
Start-Sleep -Seconds 5

# Health checks
Write-Host ""
Write-Host "Running health checks..." -ForegroundColor Blue

try {
    $apiHealth = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction Stop
    Write-Host "SUCCESS: API Server is healthy" -ForegroundColor Green
} catch {
    Write-Host "ERROR: API Server health check failed" -ForegroundColor Red
    Write-Host "Check the API server window for errors" -ForegroundColor Yellow
}

try {
    $mlHealth = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -ErrorAction Stop
    Write-Host "SUCCESS: ML Server is healthy" -ForegroundColor Green
} catch {
    Write-Host "ERROR: ML Server health check failed" -ForegroundColor Red
    Write-Host "Check the ML server window for errors" -ForegroundColor Yellow
}

# Create test patient
Write-Host ""
Write-Host "Creating test patient..." -ForegroundColor Blue
try {
    $body = @{
        patient_id = "demo_patient_001"
        sampling_interval_seconds = 60
    } | ConvertTo-Json

    Invoke-WebRequest -Uri "http://localhost:8000/patients" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -UseBasicParsing | Out-Null
    
    Write-Host "SUCCESS: Test patient created (demo_patient_001)" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Patient creation skipped (may already exist)" -ForegroundColor Yellow
}

# Display status
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "          SYSTEM STARTED SUCCESSFULLY!                          " -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Blue
Write-Host "   API Server:     http://localhost:8000" -ForegroundColor Green
Write-Host "   API Docs:       http://localhost:8000/docs" -ForegroundColor Green
Write-Host "   ML Server:      http://localhost:8001" -ForegroundColor Green
Write-Host "   ML Docs:        http://localhost:8001/docs" -ForegroundColor Green

Write-Host ""
Write-Host "Start Dashboard:" -ForegroundColor Blue
Write-Host "   streamlit run streamlit_dashboard.py" -ForegroundColor Yellow

Write-Host ""
Write-Host "Stop Services:" -ForegroundColor Blue
Write-Host "   Close the PowerShell windows" -ForegroundColor Yellow
Write-Host "   OR run: .\stop_system.ps1" -ForegroundColor Yellow

Write-Host ""
Write-Host "System is running!" -ForegroundColor Green
Write-Host ""