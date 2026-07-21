FROM python:3.11-slim

WORKDIR /app
COPY drift_worker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY drift_worker/scripts scripts
CMD ["python", "-m", "scripts.detect_drift"]
