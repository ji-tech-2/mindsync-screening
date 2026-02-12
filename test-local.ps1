# Quick Test Script - Inference Service
# Run this to quickly test if inference service works

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Inference Service" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "flaskr")) {
    Write-Host "❌ Error: flaskr directory not found" -ForegroundColor Red
    Write-Host "   Make sure you're in mindsync-model-flask directory" -ForegroundColor Yellow
    exit 1
}

# Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found" -ForegroundColor Yellow
    Write-Host "   Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "   ⚠️  IMPORTANT: Edit .env and add:" -ForegroundColor Red
    Write-Host "      - WANDB_API_KEY" -ForegroundColor Red
    Write-Host "      - GEMINI_API_KEY" -ForegroundColor Red
    Write-Host "   Then run this script again" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🐳 Starting Docker Services" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Start database and cache
Write-Host "Starting PostgreSQL and Redis..." -ForegroundColor Yellow
docker-compose up -d database cache

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
    Write-Host "   Make sure Docker is running" -ForegroundColor Yellow
    exit 1
}

Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "📥 Downloading Artifacts from W`&B" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Download artifacts
python download_artifacts.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️  W`&B download failed" -ForegroundColor Yellow
    Write-Host "   Checking if local artifacts exist..." -ForegroundColor Yellow
    
    if (-not (Test-Path "artifacts/model.pkl")) {
        Write-Host "❌ No model artifacts found!" -ForegroundColor Red
        Write-Host "   Options:" -ForegroundColor Yellow
        Write-Host "   1. Fix WANDB_API_KEY in .env and run again" -ForegroundColor Yellow
        Write-Host "   2. Train model first (in mindsync-model-training)" -ForegroundColor Yellow
        Write-Host "   3. Set SKIP_WANDB_DOWNLOAD=true and add artifacts manually" -ForegroundColor Yellow
        
        # Cleanup
        docker-compose down
        exit 1
    }
    
    Write-Host "✅ Using local artifacts" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🧪 Running Tests" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Run pytest
pytest tests/ -v --tb=short

$testResult = $LASTEXITCODE

Write-Host ""

if ($testResult -ne 0) {
    Write-Host "❌ Some tests failed!" -ForegroundColor Red
    Write-Host "   Check output above for details" -ForegroundColor Yellow
    
    # Cleanup
    docker-compose down
    exit 1
}

Write-Host "=====================================" -ForegroundColor Green
Write-Host "✅ All tests passed!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Ask if want to start Flask server
$response = Read-Host "Do you want to start Flask server for manual testing? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "🚀 Starting Flask Server" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Server will be available at:" -ForegroundColor Yellow
    Write-Host "  http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Test endpoints:" -ForegroundColor Yellow
    Write-Host "  GET  http://localhost:5000/health" -ForegroundColor Cyan
    Write-Host "  POST http://localhost:5000/predict" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    $env:FLASK_APP = "flaskr"
    $env:FLASK_ENV = "development"
    flask run
    
} else {
    Write-Host ""
    Write-Host "✅ Tests completed. Server not started." -ForegroundColor Green
    Write-Host ""
    Write-Host "To start server manually:" -ForegroundColor Yellow
    Write-Host "  `$env:FLASK_APP = 'flaskr'" -ForegroundColor Cyan
    Write-Host "  flask run" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Cleaning up Docker services..." -ForegroundColor Yellow
docker-compose down

Write-Host "✅ Done!" -ForegroundColor Green
