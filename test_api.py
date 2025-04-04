import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8002"

# Test cases in Marathi
test_cases = [
    "तुमचे काम खूप छान आहे!",  # Your work is very nice! (Positive)
    "हे फारच वाईट आहे",        # This is very bad (Negative)
    "बरं आहे असं वाटतं",       # It seems okay (Neutral)
    "आज हवामान छान आहे",       # The weather is nice today (Positive)
    "मला हा फिल्म आवडला नाही", # I didn't like this movie (Negative)
    "परीक्षेत नापास होणे दु:खद आहे", # Failing in the exam is sad (Negative)
    "श्री अंबाबाई माते की जय", # Devotional phrase (Positive)
    "जय माता दी"               # Devotional phrase (Positive)
]

def test_single_analysis():
    print("\nTesting single text analysis endpoint (/analyze):")
    for text in test_cases:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"text": text}
        )
        result = response.json()
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")

def test_batch_analysis():
    print("\nTesting batch analysis endpoint (/analyze-batch):")
    response = requests.post(
        f"{BASE_URL}/analyze-batch",
        json={"texts": test_cases}
    )
    results = response.json()["results"]
    
    for text, result in zip(test_cases, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")

def test_facebook_post_analysis():
    print("\nTesting Facebook post analysis endpoint (/analyze-fb-post):")
    # Replace with an actual Facebook post URL
    post_url = "https://www.facebook.com/example/post"
    response = requests.get(
        f"{BASE_URL}/analyze-fb-post",
        params={"url": post_url}
    )
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        test_single_analysis()
        test_batch_analysis()
        # test_facebook_post_analysis()  # Uncomment if you have a valid Facebook post URL
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Make sure it's running on port 8002.")
    except Exception as e:
        print(f"Error: {str(e)}") 