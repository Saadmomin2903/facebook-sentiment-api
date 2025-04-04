from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os

def download_model():
    print("Downloading XLM-RoBERTa model and tokenizer...")
    
    # Create model directory if it doesn't exist
    model_dir = "xlm-roberta-base-finetuned-sentiment"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
    
    # Save to disk
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    
    print("Model and tokenizer downloaded successfully!")

if __name__ == "__main__":
    download_model() 