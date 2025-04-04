# Free Deployment Guide for Facebook Sentiment Analysis API

This guide will walk you through deploying your Facebook Post Sentiment Analysis API on Render's free tier.

## Prerequisites

1. A Render account (sign up at [render.com](https://render.com))
2. A GitHub account (to host your code repository)
3. Git installed on your local machine

## Step 1: Prepare Your Code Repository

1. Create a new GitHub repository for your project
2. Push your code to GitHub:

```bash
# Initialize git repository if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Render deployment"

# Add your GitHub repo as remote
git remote add origin https://github.com/yourusername/facebook-sentiment-api.git

# Push to GitHub
git push -u origin main
```

## Step 2: Deploy on Render

### Option A: Deploy via Blueprint (render.yaml)

1. Log in to your Render dashboard
2. Click "New" and select "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file and set up your service
5. During setup, you'll be asked to provide your Facebook credentials as environment variables:
   - Set `FB_EMAIL` to your Facebook email
   - Set `FB_PASSWORD` to your Facebook password

### Option B: Manual Deployment

If the blueprint approach doesn't work, you can set up manually:

1. Log in to your Render dashboard
2. Click "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: fb-sentiment-api
   - **Environment**: Docker
   - **Plan**: Free
   - **Region**: Choose a region close to your users
   - **Branch**: main (or whatever your default branch is)
   - **Build Command**: `docker build -t fb-sentiment-api .`
5. Click "Advanced" and add the following environment variables:

   - `FB_EMAIL`: Your Facebook email
   - `FB_PASSWORD`: Your Facebook password
   - `MODEL_PATH`: `/app/models/best_marathi_sentiment_model.pth`
   - `PORT`: `10000`

6. Click "Create Web Service"

## Step 3: Handle Your Model File

Since the free tier doesn't support persistent disks, we need to handle the model file differently. Here are your options:

### Option A: Download the Model During Container Startup

1. Create a script to download the model during container startup:

```bash
#!/bin/bash
# download_model.sh
if [ ! -f "/app/models/best_marathi_sentiment_model.pth" ]; then
    echo "Downloading model file..."
    # Add your model download command here
    # Example: wget -O /app/models/best_marathi_sentiment_model.pth https://your-model-url
fi
```

2. Make the script executable:

```bash
chmod +x download_model.sh
```

3. Modify your Dockerfile to run this script before starting the application.

### Option B: Use a Smaller Model

1. Consider using a quantized or smaller version of your model
2. This will help with the memory constraints of the free tier

### Option C: Upgrade to a Paid Plan

If you need the full model:

1. Upgrade to a paid plan that supports persistent disks
2. Follow the original deployment guide with persistent storage

## Step 4: Verify Deployment

1. Once deployment is complete, your service will be available at the URL provided by Render (usually `https://fb-sentiment-api.onrender.com`)
2. Test the API by visiting the `/test-sentiment` endpoint:
   ```
   https://fb-sentiment-api.onrender.com/test-sentiment
   ```
3. If the test works, your API is ready to use!

## Using Your API

Your API will be available at:

```
https://fb-sentiment-api.onrender.com/analyze-fb-post?url=YOUR_FACEBOOK_POST_URL
```

## Limitations of the Free Tier

Be aware of these limitations:

1. **Memory Constraints**: The free tier has limited RAM (512MB), which may cause issues with large models. Your model is around 1GB in size, which may push these limits.

2. **CPU Constraints**: The free tier has limited CPU resources, which may affect scraping performance.

3. **Spin-down**: Free services will "spin down" after 15 minutes of inactivity. The first request after inactivity will take longer as the service starts up again (this is called a "cold start").

4. **Usage Limits**: The free tier includes 750 hours of usage per month.

5. **No Persistent Storage**: The free tier doesn't support persistent disks, so we need to handle the model file differently.

## Troubleshooting

### Model Loading Issues

If your model fails to load due to memory constraints, you have a few options:

1. **Quantize your model**: Convert your model to a smaller format (like int8 quantization)
2. **Upgrade to a paid plan**: Consider upgrading to a paid plan with more resources
3. **Split your service**: Run the scraper and sentiment analysis as separate services

### Facebook Scraping Issues

If Facebook starts blocking your scraper:

1. Add more randomized delays between actions
2. Use proxies (though this may require a paid Render plan)
3. Implement retry mechanisms

### Service Timeouts

If requests time out:

1. Increase the timeout value in your code
2. Consider splitting long-running operations into smaller tasks
3. Implement background processing with a queue (though this would require a paid plan)
