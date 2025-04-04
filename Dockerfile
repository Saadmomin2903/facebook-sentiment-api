FROM python:3.9-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome with minimal dependencies
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove unnecessary packages to reduce image size
RUN apt-get autoremove -y

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app/models

# Copy model download script
COPY download_model.sh /app/
RUN chmod +x /app/download_model.sh

# Copy application code
COPY . .

# Default environment variables
ENV PORT=10000
ENV MODEL_PATH=/app/models/best_marathi_sentiment_model.pth

# Set memory limit for Python
ENV PYTHONMALLOC=malloc
ENV PYTHONUNBUFFERED=1

# Start with limited resources
CMD /app/download_model.sh && python -m uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 1 --timeout-keep-alive 30