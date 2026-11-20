# Property Assistant API - runs on Render (free web service), a Hugging Face Docker Space, or any Docker host.
# Listens on $PORT when the host sets it (Render), else 7860 (HF Spaces / local).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models DATA_DIR=/app/data SESSION_DB=/tmp/sessions.db \
    HF_HOME=/tmp/hf

# HF Spaces run containers as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding + reranker models into the image so cold starts don't download them
COPY app/config.py app/config.py
RUN touch app/__init__.py && python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from app.config import EMBED_MODEL, RERANK_MODEL; \
TextEmbedding(EMBED_MODEL, cache_dir='/app/models'); \
TextCrossEncoder(RERANK_MODEL, cache_dir='/app/models'); print('models ready')"

COPY app/ app/
COPY data/ data/
RUN chown -R app:app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','7860'))"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
