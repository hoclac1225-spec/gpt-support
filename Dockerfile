FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# faiss cần libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chép toàn bộ dự án (kể cả vectors/)
COPY . .

# Render/Railway sẽ set PORT
CMD gunicorn app:app --bind 0.0.0.0:${PORT}
