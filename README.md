# Facebook Post Sentiment Analysis API

A powerful API that scrapes Facebook posts and performs sentiment analysis on Marathi text, providing insights into the emotional tone of the content.

## Features

- **Facebook Post Scraping**: Automatically scrapes posts from Facebook URLs
- **Marathi Sentiment Analysis**: Analyzes sentiment in Marathi text using a custom deep learning model
- **REST API**: Easy to use REST API with FastAPI
- **Dockerized**: Ready to deploy with Docker

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/facebook-sentiment-api.git
   cd facebook-sentiment-api
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download the sentiment model:
   [Download best_marathi_sentiment_model.pth](link-to-your-model)

   Place it in a location and set the `MODEL_PATH` environment variable to point to it.

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
FB_EMAIL=your_facebook_email
FB_PASSWORD=your_facebook_password
MODEL_PATH=./best_marathi_sentiment_model.pth
PORT=8000
```

### Running Locally

Start the API server:

```
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

Build and run with Docker:

```
docker build -t fb-sentiment-api .
docker run -p 8000:8000 -v $(pwd):/app --env-file .env fb-sentiment-api
```

## API Usage

### Analyze a Facebook Post

```
GET /analyze-fb-post?url=https://www.facebook.com/permalink.php?story_fbid=123456789
```

#### Response

```json
{
  "post_text": "आज खूप छान दिवस होता!",
  "sentiment": "positive",
  "confidence": 0.92,
  "sentiment_scores": {
    "positive": 0.92,
    "negative": 0.05,
    "neutral": 0.03
  }
}
```

### Test Sentiment Analysis

```
GET /test-sentiment
```

This endpoint provides a quick test of the sentiment analysis functionality using predefined examples.

## Deployment

For deploying to Render's free tier, see the [FREE_DEPLOYMENT_GUIDE.md](FREE_DEPLOYMENT_GUIDE.md) file.

## Model Training

The sentiment analysis model was trained on a dataset of Marathi texts. For details on the model architecture and training process, please check the documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses FastAPI for the API framework
- Selenium is used for Facebook scraping
- PyTorch for sentiment analysis model implementation
