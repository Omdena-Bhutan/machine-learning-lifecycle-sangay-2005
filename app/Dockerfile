FROM python:3.10-slim

WORKDIR /app

# Install system deps (optional speedups)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/requirements.txt
# Install PyTorch CPU wheels from the official index (more reliable than PyPI for large wheels)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --retries 20 --timeout 120 torch torchvision torchaudio
# Install remaining Python dependencies with increased timeout/retries
RUN pip install --no-cache-dir --retries 20 --timeout 120 -r /app/requirements.txt

# Copy project (only what API needs)
COPY src /app/src
COPY app/api.py /app/api.py
# The model may not exist yet during initial builds; mount at runtime or rebuild after training
# COPY models/trained /app/models/trained

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
CMD ["python", "api.py"]
