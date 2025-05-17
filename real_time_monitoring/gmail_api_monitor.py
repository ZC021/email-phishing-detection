import base64
import email
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gmail_api_monitor')


def get_gmail_service(credentials_path: str, token_path: str = 'token.json'):
    """Authenticate and return a Gmail service instance."""
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service


class GmailAPIMonitor:
    """Monitor Gmail account using Gmail API and OAuth."""

    def __init__(self, credentials_path: str, token_path: str = 'token.json', check_interval: int = 60):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.processor_thread = None
        self.email_queue = queue.Queue()
        self.callback: Callable[[Dict[str, Any]], None] | None = None
        self.service = None
        self.processed_ids = set()

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.callback = callback

    def start(self):
        if self.running:
            return
        self.running = True
        if not self.service:
            self.service = get_gmail_service(self.credentials_path, self.token_path)
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        self.processor_thread = threading.Thread(target=self._process_emails, daemon=True)
        self.processor_thread.start()
        logger.info('Started Gmail API monitoring')

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1)
        logger.info('Stopped Gmail API monitoring')

    def _poll(self):
        while self.running:
            try:
                results = self.service.users().messages().list(userId='me', maxResults=10).execute()
                messages = results.get('messages', [])
                for msg in messages:
                    msg_id = msg['id']
                    if msg_id in self.processed_ids:
                        continue
                    msg_data = self.service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
                    raw_email = base64.urlsafe_b64decode(msg_data['raw'].encode('utf-8'))
                    email_message = email.message_from_bytes(raw_email)
                    subject = email_message.get('Subject', '')
                    from_addr = email_message.get('From', '')
                    date_str = email_message.get('Date', '')
                    email_date = email.utils.parsedate_to_datetime(date_str) if date_str else datetime.utcnow()
                    body = self._get_email_body(email_message)
                    email_info = {
                        'id': msg_id,
                        'account': 'me',
                        'from': from_addr,
                        'subject': subject,
                        'date': email_date,
                        'body': body,
                        'raw_email': raw_email,
                    }
                    self.email_queue.put(email_info)
                    self.processed_ids.add(msg_id)
            except Exception as e:
                logger.error(f'Gmail API error: {e}')
            time.sleep(self.check_interval)

    def _process_emails(self):
        while self.running:
            try:
                try:
                    email_info = self.email_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if self.callback:
                    try:
                        self.callback(email_info)
                    except Exception as e:
                        logger.error(f'Callback error: {e}')
                self.email_queue.task_done()
            except Exception as e:
                logger.error(f'Email processor error: {e}')

    def _get_email_body(self, email_message) -> str:
        body = ''
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == 'text/plain' and 'attachment' not in str(part.get('Content-Disposition')):
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body += part.get_payload(decode=True).decode(charset, errors='replace')
                    except Exception:
                        body += part.get_payload(decode=False)
        else:
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode(charset, errors='replace')
            except Exception:
                body = email_message.get_payload(decode=False)
        return body
