FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user and grant ownership of the data directory.
# /app is intentionally left owned by root (read-only at runtime is fine —
# all writes go to /data via STATE_DB_PATH, HEARTBEAT_PATH, and out_dir).
RUN groupadd -r trader && useradd -r -g trader trader \
    && mkdir -p /data \
    && chown trader:trader /data

USER trader

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "scripts/forward_test.py", "--config", "config_forward_test.yaml", "--source", "live", "--mode", "testnet"]