FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Helpful system libs for PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl poppler-utils libmagic1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code + PDFs
COPY . .

# Warm the index during build (safe no-op if placeholder)
RUN python scripts/preindex.py || true

# Start app (Render injects $PORT)
CMD ["sh","-c","uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
