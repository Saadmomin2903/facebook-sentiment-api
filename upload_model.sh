#!/bin/bash

# Script to upload a model file to Render's persistent disk

echo "=========================================="
echo "   Render Model Upload Helper Script"
echo "=========================================="
echo ""

# Check if render CLI is installed
if ! command -v render &> /dev/null; then
    echo "The Render CLI is not installed."
    echo "Please install it first using: pip install render"
    exit 1
fi

# Check if logged in to Render
render whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "You are not logged in to Render."
    echo "Please log in first using: render login"
    exit 1
fi

# Get service ID
echo "Please enter your Render service ID (found in the URL of your service dashboard):"
read SERVICE_ID

if [ -z "$SERVICE_ID" ]; then
    echo "Service ID cannot be empty."
    exit 1
fi

# Get model file path
echo "Please enter the local path to your model file (e.g., ./best_marathi_sentiment_model.pth):"
read MODEL_PATH

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Invalid model file path."
    exit 1
fi

MODEL_FILENAME=$(basename "$MODEL_PATH")
REMOTE_PATH="/var/lib/render/model-storage/$MODEL_FILENAME"

echo ""
echo "Uploading model to Render..."
echo "This may take some time depending on your internet connection and the model size."
echo ""

# Upload the file
render scp "$MODEL_PATH" "$SERVICE_ID:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Model uploaded successfully to: $REMOTE_PATH"
    echo ""
    echo "You can now set your MODEL_PATH environment variable to:"
    echo "/app/models/$MODEL_FILENAME"
    echo "=========================================="
else
    echo ""
    echo "❌ Upload failed. Please check your service ID and try again."
    echo "If the issue persists, try the SFTP method described in the deployment guide."
fi 