# Start Demo Script
# This script starts both the backend and frontend servers for the Model Router demo

Write-Host "Starting Model Router Demo..." -ForegroundColor Green
Write-Host ""
Write-Host "Note: Backend will read configuration from src/backend/.env file" -ForegroundColor Yellow
Write-Host ""

# Start Backend Server
Write-Host "Starting Backend Server..." -ForegroundColor Cyan
$backendPath = Join-Path $PSScriptRoot "src\backend"
$logFile = Join-Path $backendPath "backend.log"
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; & '$pythonExe' -m uvicorn app:app --reload --port 8000 *>&1 | Tee-Object -FilePath '$logFile'"

Write-Host "Backend server starting on http://localhost:8000" -ForegroundColor Green
Write-Host ""

# Wait a moment for backend to initialize
Start-Sleep -Seconds 5

# Start Frontend Server
Write-Host "Starting Frontend Server..." -ForegroundColor Cyan
$frontendPath = Join-Path $PSScriptRoot "src\frontend"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "Set-Location '$frontendPath'; npm run dev"

Write-Host "Frontend server starting on http://localhost:3001" -ForegroundColor Green
Write-Host ""

Write-Host "=====================================" -ForegroundColor Yellow
Write-Host "Demo servers are starting!" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3001" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C in each terminal window to stop the servers." -ForegroundColor Gray
Write-Host ""
