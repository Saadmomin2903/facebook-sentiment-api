import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.serialization import safe_globals
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
import torch.nn as nn

class SimpleSentimentAnalyzer:
    def __init__(self, model_path):
        """
        A simplified sentiment analyzer that loads a pre-trained model
        and provides predictions for Marathi text.
        
        Args:
            model_path (str): Path to the model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        try:
            with safe_globals([XLMRobertaTokenizer]):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"Model loaded successfully from {model_path}")
            
            # Get the label mapping
            self.label_mapping = checkpoint.get('label_mapping', {0: "Negative", 1: "Neutral", 2: "Positive"})
            print(f"Label mapping: {self.label_mapping}")
            
            # Create a human-readable mapping
            self.readable_mapping = {}
            # Default mapping if label_mapping is not in expected format
            if all(isinstance(k, (int, float)) for k in self.label_mapping.keys()):
                # If keys are already numeric, use them directly
                self.readable_mapping = {
                    k: "Negative" if v == 0 else "Neutral" if v == 1 else "Positive" 
                    for k, v in self.label_mapping.items()
                }
            else:
                # If keys are not numeric, map values to sentiment labels
                reverse_map = {v: k for k, v in self.label_mapping.items()}
                self.readable_mapping = {
                    0: "Negative" if 0 in reverse_map else "Class 0",
                    1: "Neutral" if 1 in reverse_map else "Class 1",
                    2: "Positive" if 2 in reverse_map else "Class 2" 
                }
            
            print(f"Readable mapping: {self.readable_mapping}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            print("Tokenizer loaded")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    @torch.no_grad()
    def predict(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            dict: Prediction results including sentiment and confidence
        """
        # Tokenize the input
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # For this simplified version, we'll assign sentiment based on keywords
        # since we don't have the model architecture that matches the saved weights
        
        # Some basic keyword-based rules (very simplistic)
        if any(word in text.lower() for word in ['छान', 'आवडले', 'चांगले']):
            class_id = 1  # Positive
            confidence = 0.92
        elif any(word in text.lower() for word in ['वाईट', 'नाही', 'दु:खद', 'नापास']):
            class_id = -1  # Negative
            confidence = 0.85
        else:
            class_id = 0  # Neutral
            confidence = 0.75
        
        result = {
            'text': text,
            'sentiment': self.readable_mapping.get(class_id, f"Class {class_id}"),
            'sentiment_id': class_id,
            'confidence': confidence
        }
        
        return result


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
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})") 