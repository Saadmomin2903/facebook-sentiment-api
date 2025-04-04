from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

@app.get("/")
async def root():
    return {"message": "Facebook Sentiment Analysis API is running"}

@app.get("/test-sentiment")
async def test_sentiment():
    try:
        # Add your sentiment analysis test code here
        return {
            "status": "success",
            "message": "Sentiment analysis service is working",
            "model_path": os.getenv("MODEL_PATH", "Not set")
        }
    except Exception as e:
        logger.error(f"Error in test_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-fb-post")
async def analyze_fb_post(url: str):
    try:
        # Add your Facebook post analysis code here
        return {
            "status": "success",
            "message": "This endpoint will analyze Facebook posts",
            "url": url
        }
    except Exception as e:
        logger.error(f"Error in analyze_fb_post: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 