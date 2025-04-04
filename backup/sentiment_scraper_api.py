from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from scrapper import FacebookScraper
from simple_sentiment import SimpleSentimentAnalyzer

app = FastAPI(title="Facebook Comment Sentiment Analysis API")

# Pre-defined Facebook credentials
FB_EMAIL = "saadmomin5555@gmail.com"
FB_PASSWORD = "Saad@2903"

# Initialize sentiment analyzer
MODEL_PATH = "best_marathi_sentiment_model.pth"
sentiment_analyzer = SimpleSentimentAnalyzer(MODEL_PATH)

class PostUrlRequest(BaseModel):
    post_url: str

class CommentSentiment(BaseModel):
    author: str
    comment: str
    time: str
    sentiment: str
    confidence: float

class SentimentAnalysisResponse(BaseModel):
    post_info: dict
    comments_sentiment: List[CommentSentiment]
    metadata: dict

@app.post("/analyze-post-sentiment")
async def analyze_post_sentiment(request: PostUrlRequest):
    scraper = FacebookScraper()
    try:
        # Initialize and login with predefined credentials
        await scraper.initialize()
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        
        # Scrape the post
        post_data = await scraper.scrape_post(request.post_url)
        
        # Analyze the sentiment of each comment
        comments_with_sentiment = []
        for comment in post_data['comments']:
            # Only analyze if the comment has content
            if 'comment' in comment and comment['comment']:
                # Perform sentiment analysis
                sentiment_result = sentiment_analyzer.predict(comment['comment'])
                
                # Create a CommentSentiment object
                comment_sentiment = {
                    'author': comment.get('author', 'Unknown'),
                    'comment': comment['comment'],
                    'time': comment.get('time', ''),
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence']
                }
                comments_with_sentiment.append(comment_sentiment)
        
        # Prepare the response
        response = {
            'post_info': post_data['post'],
            'comments_sentiment': comments_with_sentiment,
            'metadata': {
                'total_comments': len(comments_with_sentiment),
                'scraped_at': post_data['metadata']['scraped_at']
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing post sentiment: {str(e)}")
    finally:
        await scraper.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 