from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from playwright.async_api import async_playwright
import json
from datetime import datetime
import uvicorn
from simple_sentiment import SimpleSentimentAnalyzer

app = FastAPI(title="Facebook Post Scraper and Sentiment Analysis API")

# Pre-defined Facebook credentials
FB_EMAIL = "saadmomin5555@gmail.com"
FB_PASSWORD = "Saad@2903"

# Initialize sentiment analyzer
MODEL_PATH = "best_marathi_sentiment_model.pth"
sentiment_analyzer = SimpleSentimentAnalyzer(MODEL_PATH)

# Add a simplified app description
app.description = """
### Facebook Post Scraper and Sentiment Analysis API

This API provides tools to:
1. Scrape Facebook posts and comments
2. Analyze the sentiment of comments in Marathi

#### Quick Start:
- Use `/analyze-fb-post?url=YOUR_FB_POST_URL` for a one-click analysis
"""

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
            print(f"Navigating to Facebook...")
            await self.page.goto('https://www.facebook.com/', timeout=60000)
            print(f"Facebook loaded")

            # Accept cookies if present
            try:
                print("Looking for cookie consent button...")
                cookie_button = await self.page.query_selector('button[data-cookiebanner="accept_button"]')
                if cookie_button:
                    await cookie_button.click()
                    print("Clicked cookie consent button")
                    await asyncio.sleep(3)
            except Exception as e:
                print(f'No cookie banner found: {str(e)}')

            # Login
            print("Filling login credentials...")
            await self.page.fill('#email', email)
            await self.page.fill('#pass', password)

            # Click login button
            print("Clicking login button...")
            await self.page.click('button[name="login"]')
            
            # Wait for navigation with extended timeout
            print("Waiting for navigation...")
            await self.page.wait_for_load_state('networkidle', timeout=60000)
            print("Navigation complete")
            await asyncio.sleep(10)

            # Check login success
            current_url = self.page.url
            print(f"Current URL: {current_url}")
            if 'checkpoint' in current_url or 'login' in current_url:
                raise Exception('Login failed - Please check credentials or handle 2FA')
            
            print("Login successful")

        except Exception as e:
            print(f"Login error: {str(e)}")
            # Take a screenshot for debugging
            try:
                await self.page.screenshot(path="login_error.png")
                print("Screenshot saved to login_error.png")
            except Exception as screenshot_error:
                print(f"Failed to take screenshot: {str(screenshot_error)}")
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
                    print(f"Error expanding comments: {e}")
                    break

                await asyncio.sleep(2)
        except Exception as e:
            print(f"Error in expand_all_comments: {e}")

    async def scrape_post(self, post_url: str) -> dict:
        try:
            print(f"Navigating to post: {post_url}")
            await self.page.goto(post_url, timeout=60000)
            print("Post page loaded")
            await self.page.wait_for_load_state('networkidle', timeout=60000)
            print("Network idle")
            await asyncio.sleep(10)
            
            print("Extracting post content...")

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
        print(f"Initializing scraper for sentiment analysis...")
        await scraper.initialize()
        
        print(f"Logging in with predefined credentials...")
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        
        # Scrape the post
        print(f"Scraping post: {request.post_url}")
        post_data = await scraper.scrape_post(request.post_url)
        print(f"Found {len(post_data['comments'])} comments to analyze")
        
        # Analyze the sentiment of each comment
        comments_with_sentiment = []
        for i, comment in enumerate(post_data['comments']):
            # Only analyze if the comment has content
            if 'comment' in comment and comment['comment']:
                print(f"Analyzing comment {i+1}/{len(post_data['comments'])}: {comment['comment'][:50]}...")
                try:
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
                    print(f"  Sentiment: {sentiment_result['sentiment']}, Confidence: {sentiment_result['confidence']:.2f}")
                except Exception as e:
                    print(f"Error analyzing comment: {str(e)}")
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
        print(f"Preparing response with {len(comments_with_sentiment)} analyzed comments")
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
        print(f"Error in sentiment analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing post sentiment: {str(e)}")
    finally:
        print("Closing scraper")
        await scraper.close()

# Add a simplified GET endpoint for easier usage
@app.get("/analyze-fb-post")
async def analyze_fb_post(url: str):
    """
    Single GET endpoint to analyze a Facebook post with a URL parameter.
    Uses predefined Facebook credentials and analyzes comment sentiment.
    
    Example: `/analyze-fb-post?url=https://www.facebook.com/sample/post`
    """
    scraper = FacebookScraper()
    try:
        print(f"Starting one-click analysis for URL: {url}")
        
        # Initialize and login with predefined credentials
        await scraper.initialize()
        await scraper.login(FB_EMAIL, FB_PASSWORD)
        
        # Scrape the post
        post_data = await scraper.scrape_post(url)
        
        # Summarize post information
        post_summary = {
            "title": post_data['post']['content'][:100] + "..." if len(post_data['post']['content']) > 100 else post_data['post']['content'],
            "author": post_data['post']['author'],
            "time": post_data['post']['time'],
            "total_comments": len(post_data['comments'])
        }
        
        # Analyze the sentiment of each comment
        comment_analysis = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Error": 0}
        
        for comment in post_data['comments']:
            if 'comment' in comment and comment['comment']:
                try:
                    # Perform sentiment analysis
                    sentiment_result = sentiment_analyzer.predict(comment['comment'])
                    
                    # Create a comment sentiment entry
                    comment_item = {
                        'author': comment.get('author', 'Unknown'),
                        'comment': comment['comment'],
                        'time': comment.get('time', ''),
                        'sentiment': sentiment_result['sentiment'],
                        'confidence': round(sentiment_result['confidence'], 2)
                    }
                    comment_analysis.append(comment_item)
                    
                    # Update sentiment counts
                    sentiment_counts[sentiment_result['sentiment']] += 1
                    
                except Exception as e:
                    # Still include the comment but with error sentiment
                    comment_item = {
                        'author': comment.get('author', 'Unknown'),
                        'comment': comment['comment'],
                        'time': comment.get('time', ''),
                        'sentiment': 'Error',
                        'confidence': 0.0
                    }
                    comment_analysis.append(comment_item)
                    sentiment_counts["Error"] += 1
        
        # Calculate sentiment distribution percentage
        total_comments = len(comment_analysis)
        sentiment_distribution = {
            "positive_percent": round(sentiment_counts["Positive"] / total_comments * 100, 1) if total_comments > 0 else 0,
            "negative_percent": round(sentiment_counts["Negative"] / total_comments * 100, 1) if total_comments > 0 else 0,
            "neutral_percent": round(sentiment_counts["Neutral"] / total_comments * 100, 1) if total_comments > 0 else 0,
        }
        
        # Prepare the comprehensive response
        response = {
            'post': post_summary,
            'sentiment_summary': {
                'counts': sentiment_counts,
                'distribution': sentiment_distribution
            },
            'comment_analysis': comment_analysis,
            'metadata': {
                'scraped_at': post_data['metadata']['scraped_at'],
                'analyzed_comments': total_comments
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in one-click analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing Facebook post: {str(e)}")
    finally:
        await scraper.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 