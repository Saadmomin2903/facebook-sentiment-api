from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
import sys
import logging
from custom_sentiment import CustomSentimentAnalyzer
from simple_sentiment import SimpleSentimentAnalyzer
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

class PostAnalysis(BaseModel):
    post_text: str
    sentiment: str
    confidence: float
    sentiment_scores: dict

def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def login_to_facebook(driver, email, password):
    try:
        driver.get("https://www.facebook.com")
        time.sleep(random.uniform(2, 4))
        
        # Accept cookies if the dialog appears
        try:
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(string(), 'Allow')]"))
            )
            cookie_button.click()
            time.sleep(random.uniform(1, 2))
        except:
            pass
        
        # Fill in login form
        email_field = driver.find_element(By.ID, "email")
        email_field.send_keys(email)
        time.sleep(random.uniform(0.5, 1.5))
        
        password_field = driver.find_element(By.ID, "pass")
        password_field.send_keys(password)
        time.sleep(random.uniform(0.5, 1.5))
        
        # Click login button
        login_button = driver.find_element(By.NAME, "login")
        login_button.click()
        time.sleep(random.uniform(3, 5))
        
        # Check for login success
        if "checkpoint" in driver.current_url:
            # Take screenshot for debugging
            driver.save_screenshot("login_error.png")
            raise Exception("Login failed - checkpoint encountered")
            
        return True
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return False

def scrape_facebook_post(driver, post_url):
    try:
        driver.get(post_url)
        time.sleep(random.uniform(3, 5))
        
        # Extract post content
        post_content = ""
        try:
            post_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-ad-preview='message']"))
            )
            post_content = post_element.text
        except:
            logger.warning("Could not find post content")
        
        return post_content
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        return None

@app.get("/")
async def root():
    return {"message": "Facebook Sentiment Analysis API is running"}

@app.get("/test-sentiment")
async def test_sentiment():
    try:
        model_path = os.getenv("MODEL_PATH", "Not set")
        analyzer = CustomSentimentAnalyzer(model_path)
        
        # Test with some sample Marathi text
        test_text = "आज खूप छान दिवस होता!"
        result = analyzer.analyze_sentiment(test_text)
        
        return {
            "status": "success",
            "message": "Sentiment analysis service is working",
            "model_path": model_path,
            "test_result": result
        }
    except Exception as e:
        logger.error(f"Error in test_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-fb-post")
async def analyze_fb_post(url: str):
    try:
        # Get Facebook credentials from environment variables
        fb_email = os.getenv("FB_EMAIL")
        fb_password = os.getenv("FB_PASSWORD")
        
        if not fb_email or not fb_password:
            raise HTTPException(status_code=500, detail="Facebook credentials not configured")
        
        # Initialize sentiment analyzer
        model_path = os.getenv("MODEL_PATH")
        analyzer = CustomSentimentAnalyzer(model_path)
        
        # Setup Selenium and login
        driver = setup_selenium()
        if not login_to_facebook(driver, fb_email, fb_password):
            driver.quit()
            raise HTTPException(status_code=401, detail="Facebook login failed")
        
        # Scrape the post
        post_content = scrape_facebook_post(driver, url)
        driver.quit()
        
        if not post_content:
            raise HTTPException(status_code=404, detail="Could not extract post content")
        
        # Analyze sentiment
        result = analyzer.analyze_sentiment(post_content)
        
        return {
            "post_text": post_content,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "sentiment_scores": result["sentiment_scores"]
        }
    except Exception as e:
        logger.error(f"Error in analyze_fb_post: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 