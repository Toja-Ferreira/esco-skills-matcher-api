########################################
# 1) Builder stage: compile wheels + fetch NLTK data
########################################
FROM python:3.10-slim AS builder

# Install build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements & build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Download only the NLTK models you actually need
RUN python -m nltk.downloader \
      punkt \
      punkt_tab \
      averaged_perceptron_tagger \
      -d /nltk_data

# Copy your app sources (so you can WCOPY modules if needed in builder)
COPY . .

########################################
# 2) Final stage: runtime only
########################################
FROM python:3.10-slim

# Tell NLTK where to look
ENV NLTK_DATA=/nltk_data

WORKDIR /app

# Copy in the wheels and install only those
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Copy the minimal NLTK data you downloaded
COPY --from=builder /nltk_data /nltk_data

# Copy your app code
COPY --from=builder /app /app

EXPOSE 10000
CMD ["python", "main.py"]
