import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Add XLMRobertaTokenizer to safe globals
torch.serialization.add_safe_globals([XLMRobertaTokenizer])

# Load your trained model and tokenizer
model_path = "best_marathi_sentiment_model.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# Load the model
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

# Load state dict and handle missing keys
state_dict = checkpoint['model_state_dict']
# Remove position_ids from state_dict if it exists
if 'roberta.embeddings.position_ids' in state_dict:
    del state_dict['roberta.embeddings.position_ids']
model.load_state_dict(state_dict, strict=False)
model.eval()

# Load fresh tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Convert label mapping to sentiment labels
sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get sentiment label using the label mapping
        predicted_idx = predictions.argmax().item()
        predicted_label = sentiment_labels[predicted_idx]
        confidence = predictions.max().item()
        
        return jsonify({
            'sentiment': predicted_label,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 