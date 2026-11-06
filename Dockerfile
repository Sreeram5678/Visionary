# Visionary — single container: FastAPI + UI + CPU inference.
# Runs on Hugging Face Spaces (Docker SDK, free CPU tier) or anywhere:
#   docker build -t visionary . && docker run -p 7860:7860 visionary
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 VISIONARY_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/Ultralytics HF_HOME=/tmp/hf

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt \
 && pip install --no-cache-dir --no-deps ultralytics==8.4.203 ultralytics-thop==2.2.5

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
