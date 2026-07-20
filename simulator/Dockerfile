FROM python:3.11-slim

WORKDIR /app
COPY simulator/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY simulator/scripts scripts
CMD ["python", "-m", "scripts.replay_holdout"]
