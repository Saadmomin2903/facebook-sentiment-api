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

## Step 2: Upload Model to GitHub Releases

1. Create a new release on GitHub:

   - Go to your repository on GitHub
   - Click on "Releases" in the right sidebar
   - Click "Create a new release"
   - Tag version: v1.0.0
   - Release title: Initial Release
   - Upload your model file: `best_marathi_sentiment_model.pth`
   - Click "Publish release"

2. Update the MODEL_URL in render.yaml:
   - Replace `yourusername` with your GitHub username
   - The URL format should be: `https://github.com/yourusername/facebook-sentiment-api/releases/download/v1.0.0/best_marathi_sentiment_model.pth`

## Step 3: Deploy on Render

### Option A: Deploy via Blueprint (render.yaml)

1. Log in to your Render dashboard
2. Click "New" and select "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file and set up your service
5. During setup, you'll be asked to provide your Facebook credentials as environment variables:
   - Set `FB_EMAIL` to your Facebook email
   - Set `FB_PASSWORD` to your Facebook password
   - The `MODEL_URL` will be automatically set from render.yaml

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
   - `MODEL_URL`: Your GitHub release URL
   - `PORT`: `10000`

6. Click "Create Web Service"

## Step 4: Verify Deployment

1. Once deployed, test the API using the health check endpoint:

   ```
   curl https://your-render-app.onrender.com/test-sentiment
   ```

2. Test the sentiment analysis endpoint:
   ```
   curl -X GET "https://your-render-app.onrender.com/analyze-fb-post?url=https://www.facebook.com/SaamTV/videos/1729089121344119"
   ```

## Troubleshooting

1. If the model fails to download:

   - Verify the MODEL_URL in render.yaml is correct
   - Check that the release is public and accessible
   - Ensure the model file name matches exactly

2. If the API fails to start:

   - Check the logs in Render dashboard
   - Verify all environment variables are set correctly
   - Ensure the model file is downloaded successfully

3. If you need to update the model:
   - Create a new release on GitHub
   - Update the MODEL_URL in render.yaml
   - Redeploy the service on Render

## Notes

- The free tier has limitations on memory and CPU usage
- The model will be downloaded each time the container starts
- Keep your Facebook credentials secure and never commit them to the repository
- Monitor your API usage to stay within free tier limits

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
