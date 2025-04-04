import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class SimpleSentimentAnalyzer:
    def __init__(self, model_path: str):
        """
        A simplified sentiment analyzer that loads a pre-trained model
        and provides predictions for Marathi text.
        
        Args:
            model_path (str): Path to the model file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> Tuple[nn.Module, AutoTokenizer]:
        """Load the model with proper error handling."""
        try:
            # First try loading with weights_only=True (PyTorch 2.6 default)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                logger.info("Model loaded successfully with weights_only=True")
            except Exception as e:
                logger.warning(f"Loading with weights_only=True failed: {e}")
                logger.info("Attempting to load with weights_only=False")
                
                # If that fails, try loading with weights_only=False
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info("Model loaded successfully with weights_only=False")

            # Extract components from checkpoint
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            config = checkpoint.get('config', {})
            
            # Initialize model with the same architecture as saved
            model = AutoModelForSequenceClassification.from_pretrained(
                "ai4bharat/indic-bert",
                num_labels=3,
                state_dict=model_state_dict
            )
            model.to(self.device)
            
            # Initialize tokenizer
            tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a single text."""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
                # Convert to sentiment labels
                sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = sentiment_map[predicted.item()]
                confidence = confidence.item()
                
                return sentiment, confidence
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for multiple texts."""
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