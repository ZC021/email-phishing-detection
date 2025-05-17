import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_COLUMNS = ['subject_length', 'body_length', 'url_count', 'suspicious_keywords']

def train(csv_path: str):
    """Train a simple logistic regression model."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df['is_phishing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
