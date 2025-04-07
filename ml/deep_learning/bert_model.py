import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import os
import json

class BertPhishingClassifier(nn.Module):
    """
    BERT-based model for phishing email classification.
    Uses pre-trained BERT embeddings to classify emails as phishing or legitimate.
    """
    def __init__(self, bert_model_name='bert-base-uncased', freeze_bert=False):
        super(BertPhishingClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # 2 classes: phishing or legitimate
        
        # Freeze BERT layers if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for padding
            
        Returns:
            logits: Raw model outputs for classification
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification layer
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        
        return logits

class BertPhishingDetector:
    """
    Wrapper class for the BERT-based phishing detection model.
    Handles tokenization, prediction, and integration with the main system.
    """
    def __init__(self, model_path=None, device=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 512  # Maximum sequence length for BERT
        
        # Set device (CPU or GPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = BertPhishingClassifier()
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def _load_model(self, model_path):
        """Load a trained model from disk"""
        model = BertPhishingClassifier()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def preprocess_text(self, email_text):
        """Preprocess and tokenize email text for BERT"""
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            email_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, email_text):
        """Predict whether an email is phishing or legitimate"""
        # Preprocess the email text
        inputs = self.preprocess_text(email_text)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Return prediction and confidence
        is_phishing = np.argmax(predictions, axis=1)[0] == 1
        confidence = predictions[0][1] if is_phishing else predictions[0][0]
        
        return {
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'probabilities': {
                'legitimate': float(predictions[0][0]),
                'phishing': float(predictions[0][1])
            }
        }
    
    def save_model(self, save_path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def train(self, train_texts, train_labels, eval_texts=None, eval_labels=None, 
              epochs=4, batch_size=16, learning_rate=2e-5):
        """Train the BERT model on email data"""
        # Set model to training mode
        self.model.train()
        
        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            # Process in batches
            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i+batch_size]
                batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.long).to(self.device)
                
                # Tokenize batch
                batch_encodings = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                batch_input_ids = batch_encodings['input_ids'].to(self.device)
                batch_attention_mask = batch_encodings['attention_mask'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Print progress
                if (i // batch_size) % 10 == 0:
                    print(f"  Batch {i // batch_size}/{len(train_texts) // batch_size}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / (len(train_texts) // batch_size)
            print(f"  Average Epoch Loss: {avg_epoch_loss:.4f}")
            
            # Evaluate if eval data is provided
            if eval_texts and eval_labels:
                eval_accuracy = self.evaluate(eval_texts, eval_labels, batch_size)
                print(f"  Evaluation Accuracy: {eval_accuracy:.4f}")
        
        # Set back to evaluation mode
        self.model.eval()
    
    def evaluate(self, eval_texts, eval_labels, batch_size=16):
        """Evaluate the model on validation/test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(eval_texts), batch_size):
                batch_texts = eval_texts[i:i+batch_size]
                batch_labels = torch.tensor(eval_labels[i:i+batch_size], dtype=torch.long).to(self.device)
                
                # Tokenize batch
                batch_encodings = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                batch_input_ids = batch_encodings['input_ids'].to(self.device)
                batch_attention_mask = batch_encodings['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(batch_input_ids, batch_attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        return accuracy

def prepare_email_for_bert(email_data):
    """Extract and prepare email content for BERT processing"""
    # If email_data is a dictionary with features
    if isinstance(email_data, dict):
        # Try to reconstruct email content from features
        content = f"Subject: {email_data.get('subject', '')}"
        
        # Add body if available
        if 'body' in email_data:
            content += f"\n\n{email_data['body']}"
        
        return content
    
    # If email_data is already a string (email content)
    return email_data

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_emails = [
        "Subject: Your account has been suspended. Click here to verify your details now!",
        "Subject: Meeting agenda for tomorrow",
        "Subject: URGENT: Your PayPal account needs verification",
        "Subject: Quarterly report is ready for review"
    ]
    sample_labels = [1, 0, 1, 0]  # 1 = phishing, 0 = legitimate
    
    # Initialize detector
    detector = BertPhishingDetector()
    
    # Quick training demo (in practice, would use much more data)
    print("Training demo model...")
    detector.train(sample_emails, sample_labels, epochs=1)
    
    # Test prediction
    test_email = "Subject: URGENT action required: Your account will be terminated unless you verify your information immediately. Click here: http://suspicious-link.com"
    result = detector.predict(test_email)
    
    print("\nPrediction results:")
    print(f"Is phishing: {result['is_phishing']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
