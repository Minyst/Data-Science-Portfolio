# ---- Multivariate Time-Series Anomaly Detection — Hugging Face Space (Docker SDK) ----
FROM python:3.12-slim

# System libs needed by lightgbm / catboost / numpy (libgomp = OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as a non-root user (uid 1000).
# Create it and make /home/user writable so caches don't crash.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Writable cache dirs for darts / torch / matplotlib / streamlit
    XDG_CACHE_HOME=/home/user/.cache \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# Install deps first for better layer caching
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# App source (데이터는 사용자가 업로드 — sample_data 는 이미지에 미포함)
COPY --chown=user app.py anomaly_core.py ./

EXPOSE 8501

# HF Spaces health check expects the app on the declared app_port (README: app_port: 8501)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
