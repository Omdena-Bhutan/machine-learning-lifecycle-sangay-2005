param(
  [string]$mode = "api",
  [int]$port = 5000
)

Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

Write-Host "Installing requirements..."
python -m pip install -r requirements.txt

Write-Host "Running unit tests..."
python -m pytest -q

if ($mode -eq "docker") {
  Write-Host "Building Docker image..."
  docker build -t sentiment-api:latest app/
  Write-Host "Running Docker container mapping host port $port -> container 5000"
  docker run -p ${port}:5000 sentiment-api:latest
} else {
  Write-Host "Starting API locally (python app/api.py)..."
  Push-Location app
  # Ensure Python can import `src` at repository root
  $env:PYTHONPATH = (Resolve-Path "..").Path
  # Pass requested port to the API via the PORT env var
  $env:PORT = $port.ToString()
  python api.py
  Pop-Location
}
