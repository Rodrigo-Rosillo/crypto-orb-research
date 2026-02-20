FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "scripts/forward_test.py", "--config", "config_forward_test.yaml", "--source", "live"]
