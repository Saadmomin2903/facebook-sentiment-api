#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p /app/models

# Check if model file exists
if [ ! -f "/app/models/best_marathi_sentiment_model.pth" ]; then
    echo "Model file not found. Please provide a URL to download the model file."
    echo "You can set the MODEL_URL environment variable to specify the download URL."
    
    if [ -z "$MODEL_URL" ]; then
        echo "Error: MODEL_URL environment variable is not set."
        echo "Please set MODEL_URL to the URL where your model file can be downloaded."
        exit 1
    fi
    
    echo "Downloading model from $MODEL_URL..."
    wget -O /app/models/best_marathi_sentiment_model.pth "$MODEL_URL"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download the model file."
        exit 1
    fi
    
    echo "Model downloaded successfully."
else
    echo "Model file already exists."
fi 