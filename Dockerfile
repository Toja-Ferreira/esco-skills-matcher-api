FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. First install requirements (including nltk)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Now download NLTK data (AFTER nltk is installed)
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger

# 4. Copy application code
COPY . .

EXPOSE 10000
CMD ["python", "main.py"]