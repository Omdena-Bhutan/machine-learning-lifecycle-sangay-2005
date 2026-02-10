# Sentiment Analysis ML Lifecycle (My Implementation Notes)

This repository is my end‑to‑end implementation of a sentiment analysis pipeline using HuggingFace Transformers, tracked with DVC, logged with MLflow, and served via a Flask API. I developed and tested this on Windows (PowerShell), and I’ve included a few Windows‑specific tips that helped me get everything running reliably.

What I built
- Data pipeline with DVC (prepare → train → evaluate stages)
- Transfer learning using DistilBERT (tokenization moved into training/eval to decouple Torch from prepare)
- Experiment tracking with MLflow (params, simple metrics, artifacts)
- Flask API for /predict and /health
- Docker image for the API (with CPU‑only PyTorch install)
- GitHub Actions workflows for train/test/deploy (configurable with repo secrets)

Project structure (key files only)
- params.yaml – training/data params
- dvc.yaml – pipeline definition
- src/
  - data_loader.py – cleans text, splits dataset, saves plain text splits
  - model.py – tokenizes in training, fine‑tunes DistilBERT, logs to MLflow, saves model.pt
  - evaluate.py – tokenizes in eval, computes accuracy/precision/recall/F1
  - inference.py – single‑text prediction (loads model state if present)
- app/
  - api.py – Flask app with /predict and /health
  - Dockerfile – installs CPU‑only torch first, then app deps
  - requirements.txt – minimal API deps
- .github/workflows/ – train/test/deploy templates

Prerequisites I used
- Python 3.10/3.11 (64‑bit)
- Git
- DVC
- MLflow
- Docker Desktop (for container build/run)

Windows‑specific setup notes
- Torch DLL error (c10.dll): I resolved this by installing CPU‑only PyTorch wheels and the Microsoft Visual C++ Redistributable (x64). For reference:
  - pip uninstall -y torch torchvision torchaudio
  - pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  - Install VC++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- PATH conflicts: I avoided mixing CUDA/NVIDIA DLLs by ensuring no CUDA paths shadowed the venv.
- PowerShell curl: I used Invoke-RestMethod or curl.exe for API testing.

Environment install
- Create and activate venv
  - python -m venv venv
  - venv\Scripts\activate
- Install project dependencies
  - pip install -r requirements.txt
- Install torch separately (CPU‑only example shown above)

DVC pipeline
- Initialize DVC (already initialized in this repo)
  - dvc init --no-scm (skip if .dvc exists)
- Provide dataset at data/raw/reviews.csv with columns review,sentiment
- Run pipeline
  - dvc repro
- Stages
  - prepare: src/data_loader.py → data/processed/train.pkl, data/processed/test.pkl (plain text, labels)
  - train: src/model.py → models/trained/model.pt, metrics.json
  - evaluate: src/evaluate.py → eval_metrics.json

MLflow usage
- Start UI locally
  - mlflow ui
  - Open http://localhost:5000 (or use a different port if occupied)
- Training logs params and basic metrics; model artifacts are saved locally and referenced in runs.

Quick functional checks
- Inference from CLI (loads base model if no trained state is present):
  - python -c "from src.inference import predict; print(predict('This movie was amazing!'))"
- API (from project root)
  - Ensure src is importable when starting the server:
    - $env:PYTHONPATH="."
    - python -m app.api
  - Health:
    - Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
  - Predict:
    - $body = @{ text = "I loved this movie" } | ConvertTo-Json
    - Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body -ContentType "application/json"

Docker build and run (CPU‑only, from project root)
- I build with the project root as context so COPY paths in Dockerfile resolve:
  - docker build -t sentiment-api:latest -f app/Dockerfile .
- If the model isn’t baked into the image, I mount it at runtime:
  - docker run -p 5000:5000 -v "$PWD/models/trained:/app/models/trained" sentiment-api:latest
- If port 5000 is in use, map another host port (e.g. 5050:5000).

GitHub Actions (CI/CD)
- .github/workflows/train.yml – runs dvc repro on push/schedule; expects DVC remote secret if used
- .github/workflows/test.yml – basic test runner (extend with real tests)
- .github/workflows/deploy.yml – container build/push on tags (configure REGISTRY_* secrets)

Troubleshooting I documented for myself
- DVC + pathspec: I pinned pathspec==0.11.2 if older DVC expected _DIR_MARK
- Torch on Windows: CPU‑only wheels + VC++ runtime fixed c10.dll issues
- Multiple servers on port 5000: I used netstat/taskkill to free the port
- PowerShell vs curl flags: Invoke-RestMethod or curl.exe instead of Git Bash syntax
- Docker torch download timeouts: I modified the Dockerfile to install torch from the PyTorch CPU index and added pip retries/timeouts

What’s left / future improvements
- Add unit tests in tests/ for data_loader, model training (smoke), and API integration
- Switch MLflow tracking_uri to a remote server if needed
- Add DVC remote (S3/GCS) and push artifacts
- Harden API logging/validation and add simple rate limiting
- Optionally enable evaluation during training for early stopping

Submission checklist (my runbook)
- [x] Data preprocessing tracked with DVC
- [x] Training with HuggingFace + MLflow logging
- [x] Evaluation metrics written to eval_metrics.json
- [x] Flask API with /predict and /health tested locally
- [x] Dockerfile builds on my machine (CPU‑only) and serves the API
- [x] Workflows scaffolded for CI/CD

Notes
- This README reflects exactly what I implemented and validated locally on Windows. Where environment‑specific steps were required (Torch DLLs, port conflicts, PowerShell requests), I added the commands I actually used so I can reproduce this later.
