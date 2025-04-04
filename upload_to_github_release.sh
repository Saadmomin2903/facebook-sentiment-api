#!/bin/bash

# Check if required tools are installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI is not installed. Please install it first:"
    echo "https://cli.github.com/"
    exit 1
fi

# Check if user is logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "Please log in to GitHub first:"
    echo "gh auth login"
    exit 1
fi

# Get repository name
echo "Enter your GitHub repository name (e.g., Saadmomin2903/facebook-sentiment-api):"
read REPO_NAME

# Get model file path
echo "Enter the path to your model file (e.g., ./best_marathi_sentiment_model.pth):"
read MODEL_PATH

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Create a new release
echo "Creating a new release..."
gh release create v1.0.0 "$MODEL_PATH" --title "Model File Release" --notes "Initial release of the Marathi sentiment analysis model"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Release created successfully!"
    echo ""
    echo "Now you can set the MODEL_URL in your Render dashboard to:"
    echo "https://github.com/$REPO_NAME/releases/download/v1.0.0/$(basename "$MODEL_PATH")"
    echo ""
    echo "Note: Make sure your repository is public or the release is accessible to Render's servers."
else
    echo "❌ Failed to create release. Please try again."
fi 