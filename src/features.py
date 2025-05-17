import email
import re
from typing import Dict

def extract_features(eml_path: str) -> Dict[str, int]:
    """Extract simple features from an .eml file."""
    with open(eml_path, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f)

    subject = msg.get('Subject', '') or ''

    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or 'utf-8'
                    body += payload.decode(charset, errors='ignore')
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or 'utf-8'
            body = payload.decode(charset, errors='ignore')
        else:
            body = msg.get_payload() or ''

    text = subject + '\n' + body

    url_pattern = re.compile(r'https?://\S+')
    keywords = ['verify', 'login', 'password', 'urgent', 'account', 'limited']

    return {
        'subject_length': len(subject),
        'body_length': len(body),
        'url_count': len(url_pattern.findall(text)),
        'suspicious_keywords': sum(1 for kw in keywords if kw.lower() in text.lower())
    }
