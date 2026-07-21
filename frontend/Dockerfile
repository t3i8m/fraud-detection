FROM python:3.11-slim

WORKDIR /app
COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/server.py .
COPY frontend/index.html .

CMD ["python", "server.py"]
