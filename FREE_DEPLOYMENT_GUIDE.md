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
   - **Build Command**: Leave blank (Docker will handle this)
   - **Start Command**: `docker run -p 10000:10000 -v /var/lib/render/model-storage:/app/models -e MODEL_PATH=/app/models/best_marathi_sentiment_model.pth -e FB_EMAIL=$FB_EMAIL -e FB_PASSWORD=$FB_PASSWORD -e PORT=10000 $RENDER_IMAGE_NAME`
5. Click "Advanced" and add the following environment variables:

   - `FB_EMAIL`: Your Facebook email
   - `FB_PASSWORD`: Your Facebook password
   - `MODEL_PATH`: `/app/models/best_marathi_sentiment_model.pth`
   - `PORT`: `10000`

6. Click "Create Web Service"

## Step 3: Set Up Persistent Disk for Your Model

The free tier of Render has a 1GB persistent disk limit, which is just enough for your model:

1. From your service dashboard, go to the "Disks" tab
2. Click "Create Disk"
3. Configure the disk:
   - **Name**: model-storage
   - **Mount Path**: `/app/models`
   - **Size**: 1GB (maximum for free tier)
4. Click "Create"

## Step 4: Upload Your Model to the Persistent Disk

There are two ways to upload your model:

### Option A: Using the Render CLI

1. Install the Render CLI:

   ```bash
   pip install render
   ```

2. Log in to Render:

   ```bash
   render login
   ```

3. Make the upload script executable:

   ```bash
   chmod +x upload_model.sh
   ```

4. Run the upload script:
   ```bash
   ./upload_model.sh
   ```
   Follow the prompts to upload your model file.

### Option B: Using SFTP (Alternative Method)

1. From your service dashboard, go to the "Shell" tab
2. Click "Connect via SFTP"
3. Use the provided credentials to connect using an SFTP client like FileZilla
4. Navigate to the `/var/lib/render/model-storage` directory
5. Upload your `best_marathi_sentiment_model.pth` file

## Step 5: Verify Deployment

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
