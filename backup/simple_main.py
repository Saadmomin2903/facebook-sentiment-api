from simple_sentiment import SimpleSentimentAnalyzer

try:
    # Initialize the sentiment analyzer
    MODEL_PATH = "best_marathi_sentiment_model.pth"
    analyzer = SimpleSentimentAnalyzer(MODEL_PATH)
    
    # Test multiple sentences
    sentences = [
        "तुमचे काम खूप छान आहे!",            # Your work is very nice!
        "हे फारच वाईट आहे",                  # This is very bad
        "बरं आहे असं वाटतं",                 # It seems okay
        "आज हवामान छान आहे",                # The weather is nice today
        "मला हा फिल्म आवडला नाही",           # I didn't like this movie
        "परीक्षेत नापास होणे दु:खद आहे"     # Failing in the exam is sad
    ]
    
    print("\n" + "="*50)
    print("MARATHI SENTIMENT ANALYSIS")
    print("="*50)
    
    # Analyze each sentence
    for sentence in sentences:
        result = analyzer.predict(sentence)
        print("\n" + "-"*50)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    print("\n" + "="*50)
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 