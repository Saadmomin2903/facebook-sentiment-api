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
import io
import mmap
import resource
from dotenv import load_dotenv
import torch
import torch.quantization
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import gc
import sys
import numpy as np
from pathlib import Path
import psutil

# Force absolute minimal resource usage
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def limit_memory():
    """Limit memory usage for the current process"""
    process = psutil.Process(os.getpid())
    
    # Set soft limit to 450MB
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (450 * 1024 * 1024, hard))
    
    # Set CPU affinity to use only one core
    try:
        process.cpu_affinity([0])
    except AttributeError:
        pass  # Not supported on all platforms

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.chunk_size = 32 * 1024 * 1024  # 32MB chunks
        
    @staticmethod
    def optimize_memory():
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
    def load_model_in_chunks(self, model_path):
        try:
            # Clear any existing model
            if self.model is not None:
                del self.model
                self.optimize_memory()

            logger.info("Loading model in chunks...")
            
            # Initialize base model with minimal memory
            model = XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base",
                num_labels=3,
                torchscript=True,
                low_cpu_mem_usage=True,
                return_dict=False
            )
            
            # Use memory mapping to load the file
            with open(model_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Read file size
                    file_size = len(mm)
                    
                    # Read in chunks using memory mapping
                    buffer = bytearray()
                    offset = 0
                    
                    while offset < file_size:
                        chunk_size = min(self.chunk_size, file_size - offset)
                        buffer.extend(mm[offset:offset + chunk_size])
                        offset += chunk_size
                        
                        logger.info(f"Read {offset}/{file_size} bytes")
                        self.optimize_memory()
                    
                    # Load checkpoint from buffer
                    checkpoint = torch.load(
                        io.BytesIO(buffer),
                        map_location='cpu',
                        weights_only=True
                    )
                    
                    # Clean up buffer
                    del buffer
                    self.optimize_memory()
            
            # Process state dict in chunks
            state_dict = checkpoint['model_state_dict']
            del checkpoint
            self.optimize_memory()
            
            if 'roberta.embeddings.position_ids' in state_dict:
                del state_dict['roberta.embeddings.position_ids']
            
            # Load state dict in chunks with numpy intermediates
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], torch.Tensor):
                    # Process tensor in chunks if it's large
                    tensor = state_dict[key]
                    if tensor.numel() * tensor.element_size() > self.chunk_size:
                        # Convert to numpy array in chunks
                        shape = tensor.shape
                        dtype = tensor.dtype
                        np_array = np.empty(shape, dtype=np.float32)
                        
                        # Process in chunks
                        chunk_size = self.chunk_size // (tensor.element_size() * np.prod(shape[1:]))
                        for i in range(0, shape[0], chunk_size):
                            end = min(i + chunk_size, shape[0])
                            np_array[i:end] = tensor[i:end].cpu().numpy()
                            self.optimize_memory()
                        
                        # Convert back to torch tensor
                        state_dict[key] = torch.from_numpy(np_array)
                        del np_array
                    else:
                        # Small tensor, process directly
                        state_dict[key] = torch.from_numpy(np.array(tensor.cpu()))
                    
                    self.optimize_memory()
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            self.optimize_memory()
            
            # Quantize and optimize
            model.eval()
            model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.LayerNorm}, 
                dtype=torch.qint8
            )
            
            # Convert to TorchScript with optimization
            with torch.jit.optimized_execution(True):
                model = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(model)
            
            self.model = model
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def initialize(self):
        try:
            if self.is_initialized:
                return
            
            logger.info("Initializing model manager...")
            
            # Set memory limits
            limit_memory()
            
            # Add tokenizer to safe globals
            torch.serialization.add_safe_globals([XLMRobertaTokenizer])
            
            # Clear memory
            self.optimize_memory()
            
            # Load model in chunks
            self.load_model_in_chunks(MODEL_PATH)
            
            # Load tokenizer with minimal memory
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                "xlm-roberta-base",
                local_files_only=True,
                low_cpu_mem_usage=True,
                do_lower_case=False
            )
            
            self.is_initialized = True
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise

    async def analyze_text(self, text: str) -> dict:
        """Analyze the sentiment of a given text."""
        try:
            if not self.is_initialized:
                self.initialize()
                
            # Tokenize with minimal memory
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
                add_special_tokens=True
            )
            
            # Get prediction with minimal memory
            with torch.no_grad():
                outputs = self.model(**inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits
                
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                predicted_idx = predictions.argmax().item()
                predicted_label = self.sentiment_labels[predicted_idx]
                confidence = float(predictions.max().item())
                
                # Clean up
                del outputs, predictions, inputs, logits
                self.optimize_memory()
                
                return {
                    "sentiment": predicted_label,
                    "confidence": confidence
                }
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze sentiment: {str(e)}"
            )

# Initialize model manager
model_manager = ModelManager()

# Initialize FastAPI app with minimal settings
app = FastAPI(
    title="Marathi Sentiment Analysis API",
    description="API for analyzing sentiment in Marathi text",
    version="1.0.0",
    docs_url=None,  # Disable Swagger UI to save memory
    redoc_url=None  # Disable ReDoc to save memory
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model manager on startup"""
    try:
        # Set process priority to low
        os.nice(19)
        # Initialize model manager
        model_manager.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize on startup: {e}")
        raise

# Update the analyze_sentiment_text function to use model manager
async def analyze_sentiment_text(text: str) -> dict:
    return await model_manager.analyze_text(text)

# Pre-defined Facebook credentials (from environment variables for security)
FB_EMAIL = os.environ.get("FB_EMAIL", "saadmomin5555@gmail.com")
FB_PASSWORD = os.environ.get("FB_PASSWORD", "Saad@2903")

# Model path from environment variables (for container deployment)
MODEL_PATH = os.environ.get("MODEL_PATH", "best_marathi_sentiment_model.pth")

# Set host and port from environment variables (for cloud deployments)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8002))

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
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-notifications',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
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
            for _ in range(10):  # Try expanding 10 times maximum
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
            await asyncio.sleep(10)
            
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
                
                commentElements.forEach(comment => {
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
        for i, comment in enumerate(post_data['comments']):
            # Only analyze if the comment has content
            if 'comment' in comment and comment['comment']:
                logger.info(f"Analyzing comment {i+1}/{len(post_data['comments'])}: {comment['comment'][:50]}...")
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
        for comment in post_data['comments']:
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
    
    # Test each sample
    for text, expected in TEST_SENTENCES.items():
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
        for text in request.texts:
            result = await analyze_sentiment_text(text)
            results.append(SentimentResponse(**result))
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT) 