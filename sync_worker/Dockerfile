FROM python:3.11-slim 

WORKDIR /app
COPY sync_worker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY sync_worker/scripts scripts
CMD ["python", "-m", "scripts.sync_profiles"]
