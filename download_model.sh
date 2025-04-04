#!/bin/bash

MODEL_DIR="/app/models"
MODEL_PATH="${MODEL_DIR}/best_marathi_sentiment_model.pth"
MODEL_URL="${MODEL_URL:-https://github.com/Saadmomin2903/facebook-sentiment-api/releases/download/v1.0.0/best_marathi_sentiment_model.pth}"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

# Create directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download model in chunks to reduce memory consumption
echo "Downloading model to $MODEL_PATH from $MODEL_URL"
wget -q --show-progress --progress=bar:force --tries=3 --waitretry=5 \
    --limit-rate=10M \
    --content-disposition \
    -O "$MODEL_PATH" \
    "$MODEL_URL" || {
    echo "Failed to download model!"
    exit 1
}

echo "Model downloaded successfully."