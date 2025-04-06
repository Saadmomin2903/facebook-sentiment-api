import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.quantization
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import gc
import os
from playwright.async_api import async_playwright
import logging
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Marathi Sentiment Analysis API",
    description="API for analyzing sentiment in Marathi text with special handling for devotional content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Initialization
MODEL_PATH = os.getenv("MODEL_PATH", "best_marathi_sentiment_model.pth")
FB_EMAIL = os.getenv("FB_EMAIL", "")
FB_PASSWORD = os.getenv("FB_PASSWORD", "")

# Pydantic models
class FacebookCredentials(BaseModel):
    email: str
    password: str
    post_url: str

class PostUrlRequest(BaseModel):
    post_url: str

class Comment(BaseModel):
    author: str
    comment: str
    timestamp: str
    reactions: int = 0
    isReply: bool = False

class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Memory optimization function
def optimize_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Model initialization function
def load_model(model_path):
    try:
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=3,
            torchscript=True,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        state_dict = checkpoint['model_state_dict']
        if 'roberta.embeddings.position_ids' in state_dict:
            del state_dict['roberta.embeddings.position_ids']
        
        model.load_state_dict(state_dict, strict=False)
        model = torch.jit.optimize_for_inference(torch.jit.script(model))
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Initialize model and tokenizer
try:
    logger.info("Starting model initialization...")
    torch.serialization.add_safe_globals([XLMRobertaTokenizer])
    optimize_memory()
    
    model = load_model(MODEL_PATH)
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "xlm-roberta-base",
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    optimize_memory()
    
    logger.info("Sentiment analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize sentiment analyzer: {e}")
    raise

# Facebook scraping class
class FacebookScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None

    async def initialize(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    # ... [Rest of the FacebookScraper class methods remain the same as in api.py]

# Sentiment analysis function
async def analyze_sentiment_text(text: str) -> dict:
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_idx = predictions.argmax().item()
            predicted_label = sentiment_labels[predicted_idx]
            confidence = float(predictions.max().item())
            
            del outputs, predictions
            optimize_memory()
            
            return {
                "sentiment": predicted_label,
                "confidence": confidence
            }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoints
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    result = await analyze_sentiment_text(request.text)
    return SentimentResponse(**result)

@app.post("/analyze-batch")
async def analyze_batch_sentiment(request: BatchSentimentRequest):
    results = []
    for text in request.texts:
        result = await analyze_sentiment_text(text)
        results.append(SentimentResponse(**result))
    return {"results": results}

@app.post("/scrape-post")
async def scrape_facebook_post(credentials: FacebookCredentials):
    scraper = FacebookScraper()
    try:
        await scraper.initialize()
        await scraper.login(credentials.email, credentials.password)
        post_data = await scraper.scrape_post(credentials.post_url)
        return post_data
    finally:
        await scraper.close()

@app.post("/analyze-post-sentiment")
async def analyze_post_sentiment(request: PostUrlRequest):
    scraper = FacebookScraper()
    try:
        await scraper.initialize()
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        post_data = await scraper.scrape_post(request.post_url)
        
        comments_with_sentiment = []
        for comment in post_data['comments']:
            if 'comment' in comment and comment['comment']:
                sentiment_result = await analyze_sentiment_text(comment['comment'])
                comment_sentiment = {
                    'author': comment.get('author', 'Unknown'),
                    'comment': comment['comment'],
                    'time': comment.get('time', ''),
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence']
                }
                comments_with_sentiment.append(comment_sentiment)
        
        return {
            'post_info': post_data['post'],
            'comments_sentiment': comments_with_sentiment,
            'metadata': {
                'total_comments': len(comments_with_sentiment),
                'scraped_at': datetime.now().isoformat()
            }
        }
    finally:
        await scraper.close()

# Gradio Interface
def analyze_text(text):
    result = asyncio.run(analyze_sentiment_text(text))
    return f"Sentiment: {result['sentiment']}\nConfidence: {result['confidence']:.2f}"

def analyze_facebook_post(url):
    result = asyncio.run(analyze_post_sentiment(PostUrlRequest(post_url=url)))
    return json.dumps(result, indent=2)

# Create Gradio interface
demo = gr.Interface(
    fn=[analyze_text, analyze_facebook_post],
    inputs=[
        gr.Textbox(label="Enter Marathi Text for Sentiment Analysis"),
        gr.Textbox(label="Enter Facebook Post URL for Analysis")
    ],
    outputs=[
        gr.Textbox(label="Sentiment Analysis Result"),
        gr.Textbox(label="Facebook Post Analysis Result")
    ],
    title="Marathi Sentiment Analysis",
    description="Analyze sentiment in Marathi text and Facebook posts"
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 