FROM python:3.11-slim

# libpq-dev + gcc needed for psycopg2-binary and some ML deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and product data
# data/products/ must be baked in — retrieval.py reads from the filesystem
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY api.py .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
