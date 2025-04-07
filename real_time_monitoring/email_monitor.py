import imaplib
import email
import time
import threading
import queue
import logging
from datetime import datetime, timedelta
import os
import json
from email.header import decode_header
from typing import List, Dict, Any, Optional, Tuple, Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('email_monitor')

class EmailMonitor:
    """
    Real-time email monitoring system that connects to email accounts
    via IMAP and scans incoming emails for phishing attempts.
    """
    def __init__(self, config_path: str = None):
        self.config = {}
        self.accounts = {}
        self.running = False
        self.threads = []
        self.email_queue = queue.Queue()
        self.processor_thread = None
        self.callback = None
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Initialize accounts from config
            if 'accounts' in self.config:
                for account in self.config['accounts']:
                    self.add_account(
                        email=account['email'],
                        password=account['password'],
                        imap_server=account.get('imap_server'),
                        imap_port=account.get('imap_port', 993),
                        use_ssl=account.get('use_ssl', True),
                        check_interval=account.get('check_interval', 60)
                    )
            
            logger.info(f"Loaded configuration with {len(self.accounts)} accounts")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to a JSON file (without passwords)"""
        try:
            # Create a copy of the config without sensitive information
            safe_config = self.config.copy()
            if 'accounts' in safe_config:
                for account in safe_config['accounts']:
                    # Replace actual password with placeholder
                    if 'password' in account:
                        account['password'] = '********'
            
            with open(config_path, 'w') as f:
                json.dump(safe_config, f, indent=4)
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def add_account(self, email: str, password: str, imap_server: str = None,
                    imap_port: int = 993, use_ssl: bool = True,
                    check_interval: int = 60) -> None:
        """Add an email account to monitor"""
        # Auto-detect IMAP server if not provided
        if not imap_server:
            imap_server = self._detect_imap_server(email)
        
        account_info = {
            'email': email,
            'password': password,
            'imap_server': imap_server,
            'imap_port': imap_port,
            'use_ssl': use_ssl,
            'check_interval': check_interval,
            'last_checked': datetime.now() - timedelta(days=1)  # Start checking from 1 day ago
        }
        
        self.accounts[email] = account_info
        
        # Update config if we started with one
        if 'accounts' in self.config:
            # Check if account already exists in config
            account_exists = False
            for i, account in enumerate(self.config['accounts']):
                if account['email'] == email:
                    self.config['accounts'][i] = account_info
                    account_exists = True
                    break
            
            if not account_exists:
                self.config['accounts'].append(account_info)
        else:
            self.config['accounts'] = [account_info]
        
        logger.info(f"Added account {email} with server {imap_server}")
    
    def remove_account(self, email: str) -> bool:
        """Remove an email account from monitoring"""
        if email in self.accounts:
            del self.accounts[email]
            
            # Update config
            if 'accounts' in self.config:
                self.config['accounts'] = [acc for acc in self.config['accounts'] 
                                          if acc['email'] != email]
            
            logger.info(f"Removed account {email}")
            return True
        return False
    
    def _detect_imap_server(self, email: str) -> str:
        """Detect IMAP server based on email domain"""
        domain = email.split('@')[-1].lower()
        
        # Common email providers
        imap_servers = {
            'gmail.com': 'imap.gmail.com',
            'outlook.com': 'outlook.office365.com',
            'hotmail.com': 'outlook.office365.com',
            'live.com': 'outlook.office365.com',
            'yahoo.com': 'imap.mail.yahoo.com',
            'aol.com': 'imap.aol.com',
            'icloud.com': 'imap.mail.me.com',
            'protonmail.com': 'imap.protonmail.ch',
            'zoho.com': 'imap.zoho.com'
        }
        
        if domain in imap_servers:
            return imap_servers[domain]
        else:
            # Default guess
            return f"imap.{domain}"
    
    def set_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback function for processing detected emails"""
        self.callback = callback
    
    def start(self) -> None:
        """Start monitoring emails in all accounts"""
        if self.running:
            logger.warning("Email monitor is already running")
            return
        
        self.running = True
        
        # Start a thread for each account
        for email_addr, account_info in self.accounts.items():
            thread = threading.Thread(
                target=self._monitor_account,
                args=(email_addr, account_info),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Start the email processor thread
        self.processor_thread = threading.Thread(
            target=self._process_emails,
            daemon=True
        )
        self.processor_thread.start()
        
        logger.info(f"Started monitoring {len(self.accounts)} email accounts")
    
    def stop(self) -> None:
        """Stop monitoring emails"""
        if not self.running:
            logger.warning("Email monitor is not running")
            return
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0)
        
        self.threads = []
        self.processor_thread = None
        
        logger.info("Stopped email monitoring")
    
    def _monitor_account(self, email_addr: str, account_info: Dict[str, Any]) -> None:
        """Monitor a single email account for new emails"""
        check_interval = account_info['check_interval']
        
        while self.running:
            try:
                # Connect to IMAP server
                if account_info['use_ssl']:
                    mail = imaplib.IMAP4_SSL(account_info['imap_server'], account_info['imap_port'])
                else:
                    mail = imaplib.IMAP4(account_info['imap_server'], account_info['imap_port'])
                
                # Login
                mail.login(email_addr, account_info['password'])
                
                # Select inbox
                mail.select('INBOX')
                
                # Get emails since last check
                since_date = account_info['last_checked'].strftime("%d-%b-%Y")
                search_criteria = f'(SINCE "{since_date}")'
                
                status, messages = mail.search(None, search_criteria)
                
                if status == 'OK':
                    message_ids = messages[0].split()
                    
                    for msg_id in message_ids:
                        status, msg_data = mail.fetch(msg_id, '(RFC822)')
                        
                        if status == 'OK':
                            raw_email = msg_data[0][1]
                            email_message = email.message_from_bytes(raw_email)
                            
                            # Extract relevant information
                            subject = self._decode_email_header(email_message['Subject'])
                            from_addr = self._decode_email_header(email_message['From'])
                            date_str = email_message['Date']
                            
                            # Convert date to datetime
                            email_date = email.utils.parsedate_to_datetime(date_str)
                            
                            # Skip if email is older than last_checked
                            if email_date <= account_info['last_checked']:
                                continue
                            
                            # Extract email body
                            body = self._get_email_body(email_message)
                            
                            # Create email info dict
                            email_info = {
                                'id': msg_id.decode(),
                                'account': email_addr,
                                'from': from_addr,
                                'subject': subject,
                                'date': email_date,
                                'body': body,
                                'raw_email': raw_email
                            }
                            
                            # Add to processing queue
                            self.email_queue.put(email_info)
                            
                            # Update last_checked
                            if email_date > account_info['last_checked']:
                                account_info['last_checked'] = email_date
                
                # Logout
                mail.logout()
                
            except Exception as e:
                logger.error(f"Error monitoring {email_addr}: {str(e)}")
            
            # Sleep until next check
            time.sleep(check_interval)
    
    def _process_emails(self) -> None:
        """Process emails from the queue"""
        while self.running:
            try:
                # Get email from queue with timeout
                try:
                    email_info = self.email_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the email if callback is set
                if self.callback:
                    try:
                        self.callback(email_info)
                    except Exception as e:
                        logger.error(f"Error in callback processing: {str(e)}")
                
                # Mark task as done
                self.email_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in email processor: {str(e)}")
    
    def _decode_email_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        
        decoded_header = decode_header(header)
        header_parts = []
        
        for content, encoding in decoded_header:
            if isinstance(content, bytes):
                if encoding:
                    header_parts.append(content.decode(encoding))
                else:
                    # Try utf-8, fallback to latin-1
                    try:
                        header_parts.append(content.decode('utf-8'))
                    except UnicodeDecodeError:
                        header_parts.append(content.decode('latin-1', errors='replace'))
            else:
                header_parts.append(content)
        
        return " ".join(header_parts)
    
    def _get_email_body(self, email_message) -> str:
        """Extract email body from email message"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Get text parts
                if content_type == "text/plain":
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        payload = part.get_payload(decode=True)
                        body += payload.decode(charset, errors='replace')
                    except:
                        body += part.get_payload(decode=False)
                
        else:
            # Not multipart - get payload directly
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode(charset, errors='replace')
            except:
                body = email_message.get_payload(decode=False)
        
        return body

# Example use case
if __name__ == "__main__":
    # Example callback function for detected emails
    def process_email(email_info):
        print(f"New email detected from {email_info['account']}:")
        print(f"From: {email_info['from']}")
        print(f"Subject: {email_info['subject']}")
        print(f"Date: {email_info['date']}")
        print(f"Body preview: {email_info['body'][:100]}...")
        print("-" * 50)
    
    # Create monitor
    monitor = EmailMonitor()
    
    # Set callback
    monitor.set_callback(process_email)
    
    # Example - Add your account (NOT RECOMMENDED to hardcode credentials)
    # monitor.add_account(
    #    email="your-email@example.com",
    #    password="your-password",
    #    check_interval=30  # Check every 30 seconds
    # )
    
    # Start monitoring
    monitor.start()
    
    try:
        # Run for 10 minutes as an example
        time.sleep(600)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop monitoring
        monitor.stop()
