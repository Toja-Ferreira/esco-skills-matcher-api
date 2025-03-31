FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Production command (single line, no backslashes)
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120"]