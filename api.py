#api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from playwright.async_api import async_playwright
import json
from datetime import datetime
import uvicorn
import logging
import os
import gc
from dotenv import load_dotenv
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Marathi Sentiment Analysis API",
    description="API for analyzing sentiment in Marathi text with special handling for devotional content",
    version="1.0.0"
)

# Pre-defined Facebook credentials (from environment variables for security)
FB_EMAIL = os.environ.get("FB_EMAIL", "saadmomin5555@gmail.com")
FB_PASSWORD = os.environ.get("FB_PASSWORD", "Saad@2903")

# Model path from environment variables (for container deployment)
MODEL_PATH = os.environ.get("MODEL_PATH", "best_marathi_sentiment_model.pth")

# Set host and port from environment variables (for cloud deployments)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8002))

# Global variables for model and tokenizer
model = None
tokenizer = None
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Lazy loader for model and tokenizer
async def get_model_and_tokenizer():
    global model, tokenizer
    
    # Only load if not already loaded
    if model is None or tokenizer is None:
        try:
            logger.info("Loading sentiment model on demand...")
            
            # Load the model with memory optimizations
            model = XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base", 
                num_labels=3,
                torchscript=True,
                low_cpu_mem_usage=True
            )
            model.eval()
            
            # Load the state dict
            logger.info(f"Loading model weights from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            
            # Extract state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Remove position_ids if it exists
            if 'roberta.embeddings.position_ids' in state_dict:
                del state_dict['roberta.embeddings.position_ids']
                
            # Load state dict with strict=False to ignore missing keys
            model.load_state_dict(state_dict, strict=False)
            
            # Apply quantization to reduce memory usage
            logger.info("Applying dynamic quantization to model")
            try:
                # Try quantizing the model to int8
                from torch.quantization import quantize_dynamic
                model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                logger.info("Model successfully quantized")
            except Exception as e:
                logger.warning(f"Quantization failed, using full precision model: {e}")
            
            # Load tokenizer
            logger.info("Loading tokenizer")
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return model, tokenizer

# Dictionary of test sentences with known sentiments for validation
TEST_SENTENCES = {
    "तुमचे काम खूप छान आहे!": "Positive",           # Your work is very nice!
    "हे फारच वाईट आहे": "Negative",                 # This is very bad
    "बरं आहे असं वाटतं": "Neutral",                 # It seems okay
    "आज हवामान छान आहे": "Positive",               # The weather is nice today
    "मला हा फिल्म आवडला नाही": "Negative",          # I didn't like this movie
    "परीक्षेत नापास होणे दु:खद आहे": "Negative",    # Failing in the exam is sad
    "श्री अंबाबाई माते की जय": "Positive",          # Devotional phrase (should be positive)
    "जय माता दी": "Positive"                        # Devotional phrase (should be positive)
}

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

class PostData(BaseModel):
    author: str
    post_content: str
    post_time: str
    post_url: str
    comments: List[dict]
    scraped_at: str

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

class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

class FacebookScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None

    async def initialize(self):
        self.playwright = await async_playwright().start()
        # Use minimal browser settings to reduce memory usage
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-notifications',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-gpu',
                '--disable-extensions',
                '--disable-software-rasterizer',
                '--disable-default-apps',
                '--mute-audio',
                '--no-zygote',
                '--single-process'
            ]
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},  # Smaller viewport
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        self.page = await self.context.new_page()

    async def login(self, email: str, password: str):
        try:
            logger.info(f"Navigating to Facebook...")
            await self.page.goto('https://www.facebook.com/', timeout=60000)
            logger.info(f"Facebook loaded")

            # Accept cookies if present
            try:
                logger.info("Looking for cookie consent button...")
                cookie_button = await self.page.query_selector('button[data-cookiebanner="accept_button"]')
                if cookie_button:
                    await cookie_button.click()
                    logger.info("Clicked cookie consent button")
                    await asyncio.sleep(3)
            except Exception as e:
                logger.info(f'No cookie banner found: {str(e)}')

            # Login
            logger.info("Filling login credentials...")
            await self.page.fill('#email', email)
            await self.page.fill('#pass', password)

            # Click login button
            logger.info("Clicking login button...")
            await self.page.click('button[name="login"]')
            
            # Wait for navigation with extended timeout
            logger.info("Waiting for navigation...")
            await self.page.wait_for_load_state('networkidle', timeout=60000)
            logger.info("Navigation complete")
            await asyncio.sleep(10)

            # Check login success
            current_url = self.page.url
            logger.info(f"Current URL: {current_url}")
            if 'checkpoint' in current_url or 'login' in current_url:
                raise Exception('Login failed - Please check credentials or handle 2FA')
            
            logger.info("Login successful")

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            # Take a screenshot for debugging
            try:
                await self.page.screenshot(path="login_error.png")
                logger.info("Screenshot saved to login_error.png")
            except Exception as screenshot_error:
                logger.error(f"Failed to take screenshot: {str(screenshot_error)}")
            raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

    async def expand_all_comments(self):
        try:
            # Limit to fewer expansion attempts to save memory
            for _ in range(5):  # Reduced from 10 to 5
                try:
                    # Click "View more comments" buttons
                    more_buttons = await self.page.query_selector_all('div[role="button"]')
                    clicked = False

                    for button in more_buttons:
                        try:
                            text = await button.text_content()
                            if text and ('view more comments' in text.lower() or 
                                       'previous comments' in text.lower()):
                                await button.click()
                                clicked = True
                                await asyncio.sleep(2)
                        except:
                            continue

                    if not clicked:
                        break

                except Exception as e:
                    logger.error(f"Error expanding comments: {e}")
                    break

                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error in expand_all_comments: {e}")

    async def scrape_post(self, post_url: str) -> dict:
        try:
            logger.info(f"Navigating to post: {post_url}")
            await self.page.goto(post_url, timeout=60000)
            logger.info("Post page loaded")
            await self.page.wait_for_load_state('networkidle', timeout=60000)
            logger.info("Network idle")
            await asyncio.sleep(5)  # Reduced from 10 to 5
            
            logger.info("Extracting post content...")

            # Get post content
            post_data = await self.page.evaluate('''() => {
                const postTexts = Array.from(document.querySelectorAll('div[dir="auto"]'))
                    .map(el => el.textContent.trim())
                    .filter(text => text.length > 0);

                const postContent = postTexts.reduce((longest, current) => 
                    current.length > longest.length ? current : longest, '');

                const authorElement = document.querySelector('h2 a') || 
                                    document.querySelector('strong a') ||
                                    document.querySelector('a[role="link"]');

                const timestampElement = Array.from(document.querySelectorAll('a[role="link"] span'))
                    .find(span => {
                        const text = span.textContent.toLowerCase();
                        return text.includes('h') || text.includes('m') || text.includes('d');
                    });

                return {
                    author: authorElement ? authorElement.textContent.trim() : 'Unknown',
                    post_content: postContent,
                    post_time: timestampElement ? timestampElement.textContent.trim() : '',
                    post_url: window.location.href
                };
            }''')

            # Expand and get comments
            await self.expand_all_comments()

            comments = await self.page.evaluate('''() => {
                const comments = [];
                const commentElements = Array.from(document.querySelectorAll('div[role="article"]'));
                
                // Limit comments to reduce memory usage
                const limitedComments = commentElements.slice(0, 50);
                
                limitedComments.forEach(comment => {
                    try {
                        const contentElement = comment.querySelector('div[dir="auto"]:not([style*="display: none"])');
                        if (!contentElement) return;

                        const content = contentElement.textContent.trim();
                        if (!content) return;

                        const authorElement = comment.querySelector('a[role="link"]:not([href*="reaction"])');
                        const timestampElement = comment.querySelector('a[role="link"] span[dir="auto"]');
                        
                        comments.push({
                            'author': authorElement ? authorElement.textContent.trim() : 'Unknown',
                            'comment': content,
                            'time': timestampElement ? timestampElement.textContent.trim() : '',
                            'reactions': 0,
                            'is_reply': !!comment.closest('div[role="article"] div[role="article"]')
                        });
                    } catch (e) {
                        console.error('Error processing comment:', e);
                    }
                });

                return comments;
            }''')

            formatted_data = {
                'post': {
                    'author': post_data['author'],
                    'content': post_data['post_content'],
                    'time': post_data['post_time'],
                    'url': post_data['post_url']
                },
                'comments': comments,
                'metadata': {
                    'total_comments': len(comments),
                    'scraped_at': datetime.now().isoformat()
                }
            }

            return formatted_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scraping post: {str(e)}")

    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        # Force garbage collection
        gc.collect()

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
        # Initialize and login with predefined credentials
        logger.info("Initializing scraper for sentiment analysis...")
        await scraper.initialize()
        
        logger.info("Logging in with predefined credentials...")
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        
        # Scrape the post
        logger.info(f"Scraping post: {request.post_url}")
        post_data = await scraper.scrape_post(request.post_url)
        logger.info(f"Found {len(post_data['comments'])} comments to analyze")
        
        # Analyze the sentiment of each comment
        comments_with_sentiment = []
        
        # Process comments in batches to reduce memory usage
        batch_size = 10
        for i in range(0, len(post_data['comments']), batch_size):
            batch = post_data['comments'][i:i+batch_size]
            
            for j, comment in enumerate(batch):
                # Only analyze if the comment has content
                if 'comment' in comment and comment['comment']:
                    logger.info(f"Analyzing comment {i+j+1}/{len(post_data['comments'])}: {comment['comment'][:50]}...")
                    try:
                        # Perform sentiment analysis
                        sentiment_result = await analyze_sentiment_text(comment['comment'])
                        
                        # Create a CommentSentiment object
                        comment_sentiment = {
                            'author': comment.get('author', 'Unknown'),
                            'comment': comment['comment'],
                            'time': comment.get('time', ''),
                            'sentiment': sentiment_result['sentiment'],
                            'confidence': sentiment_result['confidence']
                        }
                        comments_with_sentiment.append(comment_sentiment)
                        logger.info(f"  Sentiment: {sentiment_result['sentiment']}, Confidence: {sentiment_result['confidence']:.2f}")
                    except Exception as e:
                        logger.error(f"Error analyzing comment: {str(e)}")
                        # Still include the comment but with error sentiment
                        comment_sentiment = {
                            'author': comment.get('author', 'Unknown'),
                            'comment': comment['comment'],
                            'time': comment.get('time', ''),
                            'sentiment': 'Error',
                            'confidence': 0.0
                        }
                        comments_with_sentiment.append(comment_sentiment)
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Prepare the response
        logger.info(f"Preparing response with {len(comments_with_sentiment)} analyzed comments")
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
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing post sentiment: {str(e)}")
    finally:
        logger.info("Closing scraper")
        await scraper.close()

@app.get("/analyze-fb-post")
async def analyze_fb_post(url: str):
    scraper = None
    try:
        # Initialize scraper
        scraper = FacebookScraper()
        await scraper.initialize()
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        
        # Scrape the post
        logger.info("Scraping post data...")
        post_data = await scraper.scrape_post(url)
        logger.info(f"Found {len(post_data['comments'])} comments")
        
        # Analyze only the comments
        analyzed_comments = []
        
        # Process comments in batches to reduce memory usage
        batch_size = 10
        for i in range(0, len(post_data['comments']), batch_size):
            batch = post_data['comments'][i:i+batch_size]
            
            for comment in batch:
                try:
                    # Only analyze the comment text if it exists
                    if 'comment' in comment and comment['comment']:
                        logger.info(f"Analyzing comment: {comment['comment'][:50]}...")
                        sentiment_result = await analyze_sentiment_text(comment['comment'])
                        analyzed_comments.append({
                            "author": comment['author'],
                            "comment": comment['comment'],
                            "time": comment['time'],
                            "sentiment": sentiment_result["sentiment"],
                            "confidence": sentiment_result["confidence"]
                        })
                except Exception as e:
                    logger.error(f"Error analyzing comment: {str(e)}")
                    analyzed_comments.append({
                        "author": comment['author'],
                        "comment": comment['comment'],
                        "time": comment['time'],
                        "sentiment": "Error",
                        "confidence": 0.0
                    })
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Calculate sentiment distribution
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Error": 0}
        for comment in analyzed_comments:
            sentiment_counts[comment["sentiment"]] += 1
        
        total_comments = len(analyzed_comments)
        distribution = {
            "positive_percent": (sentiment_counts["Positive"] / total_comments * 100) if total_comments > 0 else 0.0,
            "negative_percent": (sentiment_counts["Negative"] / total_comments * 100) if total_comments > 0 else 0.0,
            "neutral_percent": (sentiment_counts["Neutral"] / total_comments * 100) if total_comments > 0 else 0.0
        }
        
        response = {
            "post": {
                "title": post_data['post']['content'],
                "author": post_data['post']['author'],
                "time": post_data['post']['time'],
                "total_comments": len(post_data['comments'])
            },
            "sentiment_summary": {
                "counts": sentiment_counts,
                "distribution": distribution
            },
            "comment_analysis": analyzed_comments,
            "metadata": {
                "scraped_at": datetime.now().isoformat(),
                "analyzed_comments": len(analyzed_comments)
            }
        }
        
        logger.info("Analysis complete")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_fb_post: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze post: {str(e)}")
    finally:
        if scraper:
            await scraper.close()

# Add a test endpoint to verify sentiment model is working correctly
@app.get("/test-sentiment")
async def test_sentiment():
    """Test endpoint to verify the sentiment model is working correctly"""
    results = []
    
    # Test only a subset of samples to reduce memory usage
    sample_items = list(TEST_SENTENCES.items())[:3]
    
    # Test each sample
    for text, expected in sample_items:
        result = await analyze_sentiment_text(text)
        results.append({
            "text": text,
            "expected": expected,
            "predicted": result["sentiment"],
            "confidence": round(result["confidence"], 2),
            "match": expected == result["sentiment"]
        })
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["match"])
    accuracy = round(correct / len(results) * 100, 1)
    
    return {
        "accuracy": f"{accuracy}%",
        "correct": correct,
        "total": len(results),
        "results": results
    }

@app.get("/")
async def root():
    return {
        "message": "Marathi Sentiment Analysis API",
        "status": "operational",
        "version": "1.0.0"
    }

async def analyze_sentiment_text(text: str) -> dict:
    """Analyze the sentiment of a given text."""
    try:
        # Get model and tokenizer
        model, tokenizer = await get_model_and_tokenizer()
        
        # Tokenize input with reduced max length
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get sentiment label using the label mapping
            predicted_idx = predictions.argmax().item()
            predicted_label = sentiment_labels[predicted_idx]
            confidence = float(predictions.max().item())  # Convert to float for JSON serialization
            
            logger.info(f"Analyzed text: '{text[:50]}...' -> {predicted_label} ({confidence:.2f})")
            
            return {
                "sentiment": predicted_label,
                "confidence": confidence
            }
    except Exception as e:
        logger.error(f"Error in sentiment analysis for text '{text[:50]}...': {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze sentiment: {str(e)}"
        )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = await analyze_sentiment_text(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(request: BatchSentimentRequest):
    try:
        results = []
        # Process in smaller batches to conserve memory
        batch_size = 5
        for i in range(0, len(request.texts), batch_size):
            batch = request.texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                result = await analyze_sentiment_text(text)
                batch_results.append(SentimentResponse(**result))
                
            results.extend(batch_results)
            # Force garbage collection after each batch
            gc.collect()
            
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use single worker to reduce memory usage
    uvicorn.run(app, host=HOST, port=PORT, workers=1)