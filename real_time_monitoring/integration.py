import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import threading

from real_time_monitoring.email_monitor import EmailMonitor
from real_time_monitoring.gmail_api_monitor import GmailAPIMonitor, get_gmail_service
from real_time_monitoring.database import DatabaseManager
from ml.feature_extraction import extract_features_from_email
from ml.model import PhishingDetectionModel

try:
    from ml.deep_learning.bert_model import BertPhishingDetector, prepare_email_for_bert
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring_integration')

class MonitoringService:
    """
    Integration service that connects email monitoring with phishing detection.
    Can be used standalone or integrated with Flask.
    """
    def __init__(self, config_path: str = None, model_path: str = None,
                 bert_model_path: str = None, notification_callback=None,
                 credentials_path: str = None, db_path: str = None):
        if credentials_path:
            self.monitor = GmailAPIMonitor(credentials_path)
        else:
            self.monitor = EmailMonitor(config_path)
        self.model = None
        self.bert_detector = None
        self.notification_callback = notification_callback
        self.detected_phishing = []  # List to store detected phishing emails
        self.detection_lock = threading.Lock()  # Lock for thread-safe access
        self.db = DatabaseManager(db_path) if db_path else None
        
        # Load phishing detection model
        if model_path and os.path.exists(model_path):
            try:
                import joblib
                self.model = joblib.load(model_path)
                logger.info(f"Loaded phishing model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                # Initialize a default model if loading fails
                self.model = PhishingDetectionModel()
                self.model.initialize()
        else:
            # Initialize a default model
            self.model = PhishingDetectionModel()
            self.model.initialize()
        
        # Load BERT model if available
        if BERT_AVAILABLE and bert_model_path and os.path.exists(bert_model_path):
            try:
                self.bert_detector = BertPhishingDetector(model_path=bert_model_path)
                logger.info(f"Loaded BERT model from {bert_model_path}")
            except Exception as e:
                logger.error(f"Failed to load BERT model: {str(e)}")
                self.bert_detector = None
        
        # Set callback for email processing
        self.monitor.set_callback(self.process_email)
    
    def add_account(self, email: str, password: str, imap_server: str = None,
                   imap_port: int = 993, use_ssl: bool = True,
                   check_interval: int = 60) -> bool:
        """Add an email account to monitor"""
        if isinstance(self.monitor, GmailAPIMonitor):
            logger.warning("Gmail API monitor manages the account via OAuth; 'add_account' is ignored")
            return True
        try:
            self.monitor.add_account(
                email=email,
                password=password,
                imap_server=imap_server,
                imap_port=imap_port,
                use_ssl=use_ssl,
                check_interval=check_interval
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add account {email}: {str(e)}")
            return False
    
    def remove_account(self, email: str) -> bool:
        """Remove an email account from monitoring"""
        if isinstance(self.monitor, GmailAPIMonitor):
            logger.warning("Gmail API monitor manages the account via OAuth; 'remove_account' is ignored")
            return True
        return self.monitor.remove_account(email)
    
    def start(self) -> bool:
        """Start the monitoring service"""
        try:
            self.monitor.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """Stop the monitoring service"""
        try:
            self.monitor.stop()
            return True
        except Exception as e:
            logger.error(f"Failed to stop monitoring service: {str(e)}")
            return False
    
    def process_email(self, email_info: Dict[str, Any]) -> None:
        """Process an email to detect phishing"""
        try:
            if self.db and self.db.message_exists(email_info['id']):
                return
            # Extract email content
            email_content = f"From: {email_info['from']}\nSubject: {email_info['subject']}\n\n{email_info['body']}"
            
            # Extract features for traditional model
            features = extract_features_from_email(email_content)
            
            # Make prediction with traditional model
            prediction = self.model.predict([features])
            proba = self.model.predict_proba([features])
            
            # Make prediction with BERT model if available
            bert_result = None
            if self.bert_detector:
                try:
                    bert_ready_text = prepare_email_for_bert(email_content)
                    bert_result = self.bert_detector.predict(bert_ready_text)
                except Exception as e:
                    logger.error(f"BERT prediction failed: {str(e)}")
            
            # Combine results if BERT model was used
            if bert_result:
                phishing_prob = (proba[0][1] + bert_result['probabilities']['phishing']) / 2
                is_phishing = phishing_prob > 0.5
            else:
                is_phishing = prediction[0] == 1
                phishing_prob = proba[0][1] if is_phishing else 1 - proba[0][0]
            
            # Create detection result
            detection_result = {
                'email_info': email_info,
                'features': features,
                'is_phishing': is_phishing,
                'confidence': float(phishing_prob),
                'timestamp': datetime.now().isoformat(),
                'bert_used': bert_result is not None
            }
            
            # If phishing detected, store it and notify
            if is_phishing:
                with self.detection_lock:
                    self.detected_phishing.append(detection_result)

                if self.db:
                    self.db.save_detection(email_info['id'], email_info, True, float(phishing_prob))

                # Call notification callback if provided
                if self.notification_callback:
                    try:
                        self.notification_callback(detection_result)
                    except Exception as e:
                        logger.error(f"Notification callback failed: {str(e)}")

                logger.warning(f"Phishing email detected: {email_info['subject']} (from {email_info['from']})")
            else:
                if self.db:
                    self.db.save_detection(email_info['id'], email_info, False, float(phishing_prob))
                logger.info(f"Legitimate email processed: {email_info['subject']}")
        
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
    
    def get_detected_phishing(self, limit: int = 100, clear: bool = False) -> List[Dict[str, Any]]:
        """Get list of detected phishing emails"""
        if self.db:
            return self.db.get_detections(limit)

        with self.detection_lock:
            result = list(self.detected_phishing[-limit:])
            if clear:
                self.detected_phishing = []

        return result
    
    def save_configuration(self, config_path: str) -> bool:
        """Save current configuration to file"""
        try:
            self.monitor.save_config(config_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False

# Flask integration functions
def setup_monitoring_service(app, model_path=None, bert_model_path=None):
    """
    Setup the monitoring service for a Flask application
    
    Args:
        app: Flask application
        model_path: Path to the phishing detection model
        bert_model_path: Path to the BERT model (optional)
        
    Returns:
        MonitoringService instance
    """
    if not hasattr(app, 'config'):
        raise ValueError("Invalid Flask application")
    
    # Get config path from app configuration
    config_path = os.path.join(app.config.get('CONFIG_FOLDER', ''), 'monitoring_config.json')
    
    # Get model path if not provided
    if not model_path:
        model_path = os.path.join(app.config.get('MODEL_FOLDER', ''), 'phishing_model.joblib')
    
    # Create notification callback that uses Flask's notification system
    def notification_callback(detection_result):
        # Store in app context for access in routes
        if not hasattr(app, 'phishing_notifications'):
            app.phishing_notifications = []
        
        # Add to app's notification list
        app.phishing_notifications.append(detection_result)
        
        # Limit size to prevent memory issues
        if len(app.phishing_notifications) > 1000:
            app.phishing_notifications = app.phishing_notifications[-1000:]
    
    # Create service
    credentials_path = app.config.get('GOOGLE_CREDENTIALS')
    db_path = app.config.get('DETECTION_DB', 'detections.db')

    service = MonitoringService(
        config_path=config_path,
        model_path=model_path,
        bert_model_path=bert_model_path,
        notification_callback=notification_callback,
        credentials_path=credentials_path,
        db_path=db_path,
    )
    
    # Store in app context
    app.monitoring_service = service
    
    return service

def start_monitoring(app):
    """Start monitoring for a Flask application"""
    if hasattr(app, 'monitoring_service'):
        return app.monitoring_service.start()
    return False

def stop_monitoring(app):
    """Stop monitoring for a Flask application"""
    if hasattr(app, 'monitoring_service'):
        return app.monitoring_service.stop()
    return False

# Example for standalone usage
if __name__ == "__main__":
    # Example notification callback
    def print_notification(detection_result):
        print(f"ALERT: Phishing email detected!")
        print(f"From: {detection_result['email_info']['from']}")
        print(f"Subject: {detection_result['email_info']['subject']}")
        print(f"Confidence: {detection_result['confidence'] * 100:.2f}%")
        print("-" * 50)
    
    # Create service (uses Gmail API if GOOGLE_CREDENTIALS is set)
    creds_path = os.environ.get('GOOGLE_CREDENTIALS')
    service = MonitoringService(notification_callback=print_notification,
                                credentials_path=creds_path,
                                db_path='detections.db')
    
    # Example - Add your account (NOT RECOMMENDED to hardcode credentials)
    # service.add_account(
    #     email="your-email@example.com",
    #     password="your-password",
    #     check_interval=30  # Check every 30 seconds
    # )
    
    # Start monitoring
    service.start()
    
    try:
        # Run for 10 minutes as an example
        import time
        time.sleep(600)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop monitoring
        service.stop()
