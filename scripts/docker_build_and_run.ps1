# QueryFusion Docker Build & Local Run Helper
param(
    [string]$Tag = "queryfusion:latest",
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "=== QueryFusion Docker Build & Local Run ===" -ForegroundColor Cyan

# 1. Check Docker daemon
Write-Host "--> Checking Docker daemon..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker daemon is not running. Please open Docker Desktop and wait until the engine starts, then retry."
        exit 1
    }
} catch {
    Write-Error "Failed to connect to Docker. Please ensure Docker Desktop is running."
    exit 1
}

# 2. Build Container Image
Write-Host "--> Building Docker image ($Tag)..." -ForegroundColor Yellow
docker build -t $Tag .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed."
    exit 1
}

Write-Host "--> Docker build completed successfully!" -ForegroundColor Green

# 3. Run Container
Write-Host "--> Starting container on port $Port..." -ForegroundColor Yellow
Write-Host "--> Server will be available at http://localhost:$Port" -ForegroundColor Cyan
Write-Host "--> Press Ctrl+C to stop the container.`n" -ForegroundColor DarkGray

if (Test-Path ".env") {
    Write-Host "--> Loading .env (overriding container runtime/cache paths for Linux)..." -ForegroundColor DarkGray
    docker run --rm -p "${Port}:8080" --env-file .env `
        -e "APP_ENV=production" `
        -e "PORT=8080" `
        -e "HF_HOME=/app/.cache/huggingface" `
        -e "FASTEMBED_CACHE_PATH=/app/.cache/fastembed" `
        -e "UV_CACHE_DIR=/tmp/.cache/uv" `
        -e "PIP_CACHE_DIR=/tmp/.cache/pip" `
        -e "TEMP=/tmp" `
        -e "TMP=/tmp" `
        $Tag
} else {
    Write-Warning "No .env file found. Running without .env environment variables."
    docker run --rm -p "${Port}:8080" -e "APP_ENV=production" -e "PORT=8080" $Tag
}

