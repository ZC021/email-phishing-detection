import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model(df, model, test_size=0.2, random_state=42):
    """Train the phishing detection model"""
    # Ensure the dataframe has the required columns
    required_columns = ['is_phishing'] + model.feature_names
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Split features and target
    X = df[model.feature_names]
    y = df['is_phishing']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def prepare_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    # This is just a placeholder for a real dataset
    # In a real scenario, this would be replaced with actual data
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'subject_length': np.random.randint(10, 100, n_samples),
        'body_length': np.random.randint(100, 10000, n_samples),
        'reply_to_different': np.random.randint(0, 2, n_samples),
        'url_count': np.random.randint(0, 20, n_samples),
        'suspicious_tld': np.random.randint(0, 2, n_samples),
        'ip_url': np.random.randint(0, 2, n_samples),
        'url_length_avg': np.random.randint(20, 200, n_samples),
        'domain_age_days': np.random.randint(1, 3650, n_samples),
        'contains_form': np.random.randint(0, 2, n_samples),
        'contains_script': np.random.randint(0, 2, n_samples),
        'contains_iframe': np.random.randint(0, 2, n_samples),
        'suspicious_keywords': np.random.randint(0, 10, n_samples),
        'exclamation_count': np.random.randint(0, 15, n_samples),
        'urgency_count': np.random.randint(0, 8, n_samples),
        'misspelled_domain': np.random.randint(0, 2, n_samples),
        'security_words_count': np.random.randint(0, 10, n_samples),
    }
    
    # Generate target with bias towards certain features
    features = pd.DataFrame(data)
    probability = 0.1 + \
                 0.2 * features['suspicious_tld'] + \
                 0.2 * features['ip_url'] + \
                 0.1 * features['contains_form'] + \
                 0.1 * features['misspelled_domain'] + \
                 0.05 * (features['suspicious_keywords'] > 3) + \
                 0.05 * (features['urgency_count'] > 3) + \
                 0.1 * features['reply_to_different']
    
    # Clip probability to 0-1 range
    probability = np.clip(probability, 0, 1)
    
    # Generate target based on probability
    is_phishing = np.random.binomial(1, probability)
    
    # Add target to features
    features['is_phishing'] = is_phishing
    
    return features

if __name__ == "__main__":
    # This can be used to generate a sample dataset for testing
    sample_df = prepare_sample_dataset()
    sample_df.to_csv('sample_phishing_dataset.csv', index=False)
    print(f"Sample dataset created with {len(sample_df)} examples")
