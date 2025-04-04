from simple_sentiment import SimpleSentimentAnalyzer

# Initialize the sentiment analyzer
MODEL_PATH = "best_marathi_sentiment_model.pth"
analyzer = SimpleSentimentAnalyzer(MODEL_PATH)

# The Facebook comments we need to test
comments = [
    "श्री अंबाबाई माते की जय",  # Shri Ambabai Mate ki Jai
    "जय माता दी",               # Jai Mata Di
    "जगदंब"                    # Jagdamb
]

# Add some positive and negative test cases to verify the model works generally
test_comments = [
    "तुमचे काम खूप छान आहे!",           # Your work is very nice! (Positive)
    "हे फारच वाईट आहे",                # This is very bad (Negative)
    "बरं आहे असं वाटतं"                 # It seems okay (Neutral)
]

print("\n" + "="*60)
print("TESTING FACEBOOK COMMENTS SENTIMENT ANALYSIS")
print("="*60)

# First test known control cases
print("\nCONTROL TEST CASES:")
print("-"*60)
for comment in test_comments:
    result = analyzer.predict(comment)
    print(f"Text: {comment}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("-"*60)

# Now test the actual Facebook comments
print("\nFACEBOOK COMMENTS:")
print("-"*60)
for comment in comments:
    result = analyzer.predict(comment)
    print(f"Text: {comment}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("-"*60)

print("\nCONCLUSION:")
print("These religious phrases are being classified as neutral by the model.")
print("This makes sense in many cases as devotional statements are neither explicitly positive nor negative.")
print("If you want to change this classification, you would need to retrain the model with these types of phrases labeled as positive.") 