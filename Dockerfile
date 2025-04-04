FROM python:3.9-slim

WORKDIR /app

# Install Chrome and required dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver (using a fixed version for stability)
RUN CHROME_DRIVER_VERSION="114.0.5735.90" \
    && wget -q "https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip \
    && mv chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm chromedriver_linux64.zip

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app/models

# Copy application code and scripts
COPY . .
COPY download_model.sh /app/
RUN chmod +x /app/download_model.sh

# Default env variables (can be overridden)
ENV PORT=10000
ENV MODEL_PATH=/app/models/best_marathi_sentiment_model.pth

# Run the download script and then start the application
CMD /app/download_model.sh && uvicorn api:app --host 0.0.0.0 --port $PORT 