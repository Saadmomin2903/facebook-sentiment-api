# Deployment Guide for Facebook Post Sentiment Analysis API

This guide explains how to deploy the Facebook Post Sentiment Analysis API to make it globally accessible.

## Option 1: Cloud Deployment (AWS)

### Prerequisites

- AWS account
- Docker installed on your local machine
- AWS CLI configured on your local machine

### Step 1: Create an ECR Repository

1. Go to AWS ECR (Elastic Container Registry)
2. Create a new private repository for your application

```bash
aws ecr create-repository --repository-name facebook-sentiment-api
```

### Step 2: Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

# Build the Docker image
docker build -t facebook-sentiment-api .

# Tag the image
docker tag facebook-sentiment-api:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/facebook-sentiment-api:latest

# Push the image
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/facebook-sentiment-api:latest
```

### Step 3: Upload Model to S3

Since the model is large (1GB), it's better to store it in S3 and download it during container initialization:

```bash
# Create S3 bucket (if needed)
aws s3 mb s3://facebook-sentiment-models

# Upload model
aws s3 cp best_marathi_sentiment_model.pth s3://facebook-sentiment-models/
```

### Step 4: Deploy with ECS or EKS

#### Using ECS (Elastic Container Service)

1. Create an ECS cluster
2. Define a task definition with your container
3. Include a startup script to download the model from S3
4. Create a service with the task definition
5. Set up an Application Load Balancer

Add this to your task definition:

```json
{
  "containerDefinitions": [
    {
      "name": "facebook-sentiment-api",
      "image": "<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/facebook-sentiment-api:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8002,
          "hostPort": 8002
        }
      ],
      "environment": [
        {
          "name": "MODEL_BUCKET",
          "value": "facebook-sentiment-models"
        },
        {
          "name": "MODEL_KEY",
          "value": "best_marathi_sentiment_model.pth"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/facebook-sentiment-api",
          "awslogs-region": "<your-region>",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "entryPoint": ["sh", "-c"],
      "command": [
        "aws s3 cp s3://$MODEL_BUCKET/$MODEL_KEY /app/models/ && python fixed_api.py"
      ]
    }
  ]
}
```

### Step 5: Set Up a Domain and SSL

1. Register a domain with Route 53 (or use an existing domain)
2. Create a certificate with ACM (AWS Certificate Manager)
3. Configure your ALB with the certificate
4. Create a Route 53 record pointing to your ALB

## Option 2: Platform as a Service (Render, Digital Ocean App Platform)

For a simpler deployment:

### Using Render

1. Create an account on Render
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the build and start commands
5. Add a volume mount for the model
6. Deploy

Configuration example:

```yaml
# render.yaml
services:
  - type: web
    name: facebook-sentiment-api
    env: docker
    buildCommand: docker build -t facebook-sentiment-api .
    startCommand: docker run -p 8002:8002 -v /render/volume:/app/models facebook-sentiment-api
    disk:
      name: model-storage
      mountPath: /render/volume
      sizeGB: 10
```

Before deployment, upload your model file to the persistent disk.

## Option 3: Self-Hosted VPS

### Using a VPS provider (DigitalOcean, Linode, etc.)

1. Create a VPS (minimum 2GB RAM, 2 CPU cores, 30GB SSD)
2. Install Docker and Docker Compose
3. Upload your files including the model
4. Run with Docker Compose

```bash
# docker-compose.yml
version: '3'
services:
  api:
    build: .
    ports:
      - "8002:8002"
    volumes:
      - ./best_marathi_sentiment_model.pth:/app/models/best_marathi_sentiment_model.pth
    restart: always
```

4. Set up a domain name pointing to your server
5. Configure Nginx as a reverse proxy with SSL (using Let's Encrypt)

```nginx
# /etc/nginx/sites-available/facebook-sentiment-api
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Option 4: Using a Model Hosting Service

### Using Hugging Face Spaces or similar

1. Modify your code to use a Hugging Face hosted model
2. Deploy your API service separately

## Recommended Approach

For your specific use case with a large ML model, the recommended approach is **Option 1 (AWS)** or **Option 3 (Self-Hosted VPS)** because:

1. You need to handle a large model file (1GB)
2. You need control over the environment for Playwright
3. You need stable performance for both scraping and ML inference

## Production Considerations

### Model Loading Optimization

Update your fixed_api.py to load the model from the correct location:

```python
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/best_marathi_sentiment_model.pth")
```

### Environment Variables

Store sensitive information (Facebook credentials) as environment variables:

```python
FB_EMAIL = os.environ.get("FB_EMAIL", "saadmomin5555@gmail.com")
FB_PASSWORD = os.environ.get("FB_PASSWORD", "Saad@2903")
```

### API Authentication

Add authentication to your API for security:

```python
from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException, status

API_KEY = os.environ.get("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

@app.get("/analyze-fb-post", dependencies=[Security(get_api_key)])
async def analyze_fb_post(url: str):
    # Your existing code
```

### Rate Limiting

Add rate limiting to prevent abuse:

```python
from fastapi import Request
import time

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, calls=10, period=60):
        self.calls = calls
        self.period = period
        self.records = {}

    async def check(self, client_id):
        now = time.time()
        if client_id not in self.records:
            self.records[client_id] = []

        # Clean old records
        self.records[client_id] = [t for t in self.records[client_id] if now - t < self.period]

        # Check if too many requests
        if len(self.records[client_id]) >= self.calls:
            return False

        # Add new record
        self.records[client_id].append(now)
        return True

limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host
    if not await limiter.check(client_id):
        raise HTTPException(status_code=429, detail="Too many requests")
    return await call_next(request)
```

### Monitoring and Logging

Set up proper logging and monitoring using CloudWatch (AWS) or similar services.
