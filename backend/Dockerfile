# ===============================
# Base Image (Lightweight, CPU)
# ===============================
FROM python:3.11-slim

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME="/app/models/torch" \
    HF_HOME="/app/models/huggingface" \
    TRANSFORMERS_CACHE="/app/models/huggingface" \
    TRANSFORMERS_OFFLINE=0 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Create venv & install dependencies
# -------------------------------
WORKDIR /app
COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu \
        torchaudio==2.3.1+cpu \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    /opt/venv/bin/pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt && \
    rm -rf ~/.cache/pip

# -------------------------------
# Copy Project Files
# -------------------------------
COPY app/ ./app/
COPY data/ ./data/

# -------------------------------
# Security & Permissions
# -------------------------------
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /tmp/video_processing /app/logs $TORCH_HOME $HF_HOME && \
    chown -R appuser:appuser /app /tmp/video_processing /app/models

USER appuser

# -------------------------------
# Healthcheck + Port + CMD
# -------------------------------
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

COPY gunicorn_conf.py .

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
