FROM python:3.9-slim

# Set environment variables for Python and memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1
ENV PYTORCH_JIT=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Set memory limit for the container
ENV MEMORY_LIMIT=512m

WORKDIR /app

# Install Chrome and required dependencies with cleanup in same layer
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    default-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Chrome with cleanup in same layer
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install ChromeDriver with cleanup
RUN CHROME_DRIVER_VERSION="114.0.5735.90" \
    && wget -q "https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip \
    && mv chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm chromedriver_linux64.zip

# Copy requirements and install with pip optimization
COPY requirements.txt .
RUN pip install --no-cache-dir --compile -r requirements.txt \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyc" | xargs rm -rf \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyo" | xargs rm -rf \
    && pip cache purge

# Create directory for models and cache
RUN mkdir -p /app/models /app/models/cache /app/models/torch

# Copy application code and scripts
COPY . .
COPY download_model.sh /app/
RUN chmod +x /app/download_model.sh

# Default env variables
ENV PORT=10000
ENV MODEL_PATH=/app/models/best_marathi_sentiment_model.pth
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV TORCH_HOME=/app/models/torch
ENV TRANSFORMERS_OFFLINE=1

# Create startup script with memory optimization
RUN echo '#!/bin/bash\n\
    ulimit -v 512000\n\
    /app/download_model.sh\n\
    rm -rf /root/.cache/pip\n\
    rm -rf /tmp/*\n\
    python3 -c "import torch; torch.backends.cudnn.enabled=False"\n\
    exec uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 1\n' > /app/start.sh \
    && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 