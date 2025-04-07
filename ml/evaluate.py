import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained model.
    
    Args:
        model: Trained PhishingDetectionModel instance
        X_test: Test features (DataFrame or feature dictionaries)
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Calculate ROC curve and AUC if we have probability predictions
    if y_proba is not None and y_proba.shape[1] > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        metrics['avg_precision'] = average_precision_score(y_test, y_proba[:, 1])
    
    # Get feature importance if available
    if hasattr(model, 'get_feature_importance'):
        metrics['feature_importance'] = model.get_feature_importance()
    
    return metrics

def evaluate_model_from_file(model_path, test_data_path):
    """
    Load a model and test data from files and evaluate model performance.
    
    Args:
        model_path: Path to the saved model file
        test_data_path: Path to the test data CSV file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    
    # Prepare features and labels
    X_test = test_data[model.feature_names]
    y_test = test_data['is_phishing']
    
    # Evaluate the model
    return evaluate_model(model, X_test, y_test)

def plot_confusion_matrix(cm, classes=None, normalize=False, title='Confusion matrix', 
                         cmap=plt.cm.Blues, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix to plot
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map
        save_path: Path to save the figure, if None the figure is shown
    """
    if classes is None:
        classes = ['Legitimate', 'Phishing']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plot the ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        save_path: Path to save the figure, if None the figure is shown
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random model line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(precision, recall, avg_precision, save_path=None):
    """
    Plot the precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        save_path: Path to save the figure, if None the figure is shown
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_importance, top_n=10, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to display
        save_path: Path to save the figure, if None the figure is shown
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Extract names and scores
    names = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), scores, align='center')
    plt.yticks(range(len(names)), names)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    from ml.model import PhishingDetectionModel
    from ml.train import prepare_sample_dataset
    
    # Create a sample dataset
    sample_data = prepare_sample_dataset()
    
    # Initialize a model
    model = PhishingDetectionModel()
    model.initialize()
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X = sample_data.drop('is_phishing', axis=1)
    y = sample_data['is_phishing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print the results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot the confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
