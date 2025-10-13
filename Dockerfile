FROM python:3.11-slim

# libs hệ thống cho numpy / faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# dùng PORT do platform cấp; nếu không có, mặc định 8080
CMD ["bash","-lc","exec gunicorn -w 1 -k gthread -b 0.0.0.0:${PORT:-8080} app:app --threads 8 --timeout 120"]
