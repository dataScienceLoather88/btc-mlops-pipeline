# Stage 1: Builder
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/btc_forecaster.pkl

WORKDIR /app

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser src/ /app/src/

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--chdir", "/app/app", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
