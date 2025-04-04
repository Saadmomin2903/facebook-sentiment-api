import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple

class SimpleSentimentAnalyzer:
    def __init__(self, model_path: str):
        """
        A simplified sentiment analyzer that loads a pre-trained model
        and provides predictions for Marathi text.
        
        Args:
            model_path (str): Path to the model file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = self._load_model(model_path)
            print("Model loaded successfully")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
            print("Tokenizer loaded")
            
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the PyTorch model from the given path."""
        try:
            # Try loading with map_location to handle CPU/GPU compatibility
            model = torch.load(model_path, map_location=self.device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a single text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            return sentiment_map[predicted_class], confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for a batch of texts."""
        return [self.predict(text) for text in texts]


# Example usage
if __name__ == "__main__":
    MODEL_PATH = "best_marathi_sentiment_model.pth"
    
    # Initialize analyzer
    analyzer = SimpleSentimentAnalyzer(MODEL_PATH)
    
    # Example sentences
    sentences = [
        "तुमचे काम खूप छान आहे!",  # Your work is very nice!
        "हे फारच वाईट आहे",         # This is very bad
        "बरं आहे असं वाटतं"         # It seems okay
    ]
    
    # Make predictions
    for sentence in sentences:
        result = analyzer.predict(sentence)
        print("\n" + "="*50)
        print(f"Text: {sentence}")
        print(f"Sentiment: {result[0]} (Confidence: {result[1]:.2f})") 