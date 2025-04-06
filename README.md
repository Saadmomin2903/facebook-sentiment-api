# Marathi Sentiment Analysis API

This is a FastAPI and Gradio-based application for analyzing sentiment in Marathi text and Facebook posts. The application provides both API endpoints and a web interface.

## Features

- Marathi text sentiment analysis
- Facebook post sentiment analysis
- Batch sentiment analysis
- Web interface using Gradio
- RESTful API endpoints

## API Endpoints

- `/analyze` - Analyze sentiment of a single Marathi text
- `/analyze-batch` - Analyze sentiment of multiple texts
- `/scrape-post` - Scrape a Facebook post
- `/analyze-post-sentiment` - Analyze sentiment of Facebook post comments

## Environment Variables

```bash
FB_EMAIL=your_facebook_email
FB_PASSWORD=your_facebook_password
MODEL_PATH=path_to_your_model.pth
```

## Model

The model file (`best_marathi_sentiment_model.pth`) should be placed in the root directory or specified using the `MODEL_PATH` environment variable.

## Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face:

   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Name your space and set it to "Public"

2. Upload your model:

   - Go to your Space's "Files" tab
   - Upload your `best_marathi_sentiment_model.pth` file

3. Set environment variables:

   - Go to your Space's "Settings" tab
   - Add your Facebook credentials as secrets

4. The application will automatically deploy when you push your code.

## Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

The application will be available at `http://localhost:7860`

## API Usage Examples

### Analyze Single Text

```python
import requests

response = requests.post(
    "http://localhost:7860/analyze",
    json={"text": "तुमचे काम खूप छान आहे!"}
)
print(response.json())
```

### Analyze Facebook Post

```python
response = requests.post(
    "http://localhost:7860/analyze-post-sentiment",
    json={"post_url": "https://www.facebook.com/..."}
)
print(response.json())
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses FastAPI for the API framework
- Selenium is used for Facebook scraping
- PyTorch for sentiment analysis model implementation
