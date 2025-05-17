import joblib
import pandas as pd
from .features import extract_features

FEATURE_COLUMNS = ['subject_length', 'body_length', 'url_count', 'suspicious_keywords']

def predict(model_path: str, eml_path: str):
    """Load a model and predict phishing for a single email."""
    model = joblib.load(model_path)
    features = extract_features(eml_path)
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {'is_phishing': bool(pred), 'confidence': float(prob)}
