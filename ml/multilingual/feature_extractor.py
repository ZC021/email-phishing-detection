import re
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import email
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json

# Import local modules
from ml.multilingual.language_detector import LanguageDetector
from ml.multilingual.translator import Translator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multilingual_feature_extractor')

class MultilingualFeatureExtractor:
    """
    Extract features from emails in multiple languages for phishing detection.
    Supports automatic language detection and translation when needed.
    """
    
    def __init__(self, suspicious_keywords_path: str = None, 
                 translate_to_english: bool = True):
        # Initialize language detector
        self.lang_detector = LanguageDetector()
        
        # Initialize translator if needed
        self.translator = Translator() if translate_to_english else None
        self.translate_to_english = translate_to_english
        
        # Load suspicious keywords in different languages
        self.suspicious_keywords = self._load_suspicious_keywords(suspicious_keywords_path)
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.country', '.stream',
            '.download', '.win', '.bid', '.icu', '.recipe', '.racing', '.gdn', '.review'
        ]
    
    def _load_suspicious_keywords(self, file_path: Optional[str]) -> Dict[str, List[str]]:
        """Load suspicious keywords from file or use defaults"""
        # Default English keywords
        default_keywords = {
            'en': [
                'urgent', 'verify', 'account', 'banking', 'security', 'update', 'alert', 'suspended',
                'confirm', 'password', 'login', 'unusual activity', 'access', 'expire', 'authenticate',
                'unauthorized', 'fraud', 'suspicious', 'sensitive', 'statement', 'validate', 'immediately',
                'deactivate', 'confirm identity', 'limited', 'attention required'
            ]
        }
        
        # Add default keywords for other languages
        default_keywords.update({
            # German
            'de': [
                'dringend', 'bestätigen', 'konto', 'banking', 'sicherheit', 'aktualisieren', 'alarm',
                'gesperrt', 'passwort', 'anmelden', 'ungewöhnliche aktivität', 'zugriff', 'ablaufen',
                'authentifizieren', 'unbefugt', 'betrug', 'verdächtig', 'sensibel', 'sofort',
                'deaktivieren', 'identität bestätigen', 'begrenzt', 'aufmerksamkeit erforderlich'
            ],
            # French
            'fr': [
                'urgent', 'vérifier', 'compte', 'bancaire', 'sécurité', 'mettre à jour', 'alerte',
                'suspendu', 'confirmer', 'mot de passe', 'connexion', 'activité inhabituelle',
                'accès', 'expirer', 'authentifier', 'non autorisé', 'fraude', 'suspect', 'sensible',
                'relevé', 'valider', 'immédiatement', 'désactiver', 'confirmer identité', 'limité'
            ],
            # Spanish
            'es': [
                'urgente', 'verificar', 'cuenta', 'bancario', 'seguridad', 'actualizar', 'alerta',
                'suspendido', 'confirmar', 'contraseña', 'iniciar sesión', 'actividad inusual',
                'acceso', 'expirar', 'autenticar', 'no autorizado', 'fraude', 'sospechoso',
                'sensible', 'extracto', 'validar', 'inmediatamente', 'desactivar', 'confirmar identidad'
            ],
            # Italian
            'it': [
                'urgente', 'verificare', 'account', 'bancario', 'sicurezza', 'aggiornare', 'avviso',
                'sospeso', 'confermare', 'password', 'accesso', 'attività insolita', 'scadere',
                'autenticare', 'non autorizzato', 'frode', 'sospetto', 'sensibile', 'immediato',
                'disattivare', 'confermare identità', 'limitato', 'attenzione richiesta'
            ]
        })
        
        # Try to load from file if provided
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    custom_keywords = json.load(f)
                
                # Merge with defaults, allowing custom keywords to override
                for lang, keywords in custom_keywords.items():
                    if lang in default_keywords:
                        # Add new keywords without duplicates
                        default_keywords[lang] = list(set(default_keywords[lang] + keywords))
                    else:
                        default_keywords[lang] = keywords
                
                logger.info(f"Loaded suspicious keywords for {len(custom_keywords)} languages")
            except Exception as e:
                logger.error(f"Error loading suspicious keywords: {str(e)}")
        
        return default_keywords
    
    def extract_features(self, email_text: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract features from email text for phishing detection with language support.
        
        Args:
            email_text: Email content as string or bytes
            
        Returns:
            Dictionary of extracted features
        """
        # Convert bytes to string if needed
        if isinstance(email_text, bytes):
            email_text = email_text.decode('utf-8', errors='replace')
        
        # Parse the email
        try:
            parsed_email = email.message_from_string(email_text)
        except Exception as e:
            logger.error(f"Error parsing email: {str(e)}")
            # Create a minimal message if parsing fails
            parsed_email = email.message.Message()
            parsed_email.set_payload(email_text)
        
        # Extract headers
        subject = self._decode_header(parsed_email.get('Subject', ''))
        from_address = self._decode_header(parsed_email.get('From', ''))
        to_address = self._decode_header(parsed_email.get('To', ''))
        reply_to = self._decode_header(parsed_email.get('Reply-To', ''))
        
        # Extract email body
        body = self._get_email_body(parsed_email)
        
        # Detect language of subject and body
        subject_lang_info = self.lang_detector.detect_language(subject)
        body_lang_info = self.lang_detector.detect_language(body)
        
        # Use most confident language detection result, preferring body over subject
        if body_lang_info['confidence'] > 0.5:
            primary_lang = body_lang_info['language']
            language_confidence = body_lang_info['confidence']
        elif subject_lang_info['confidence'] > 0.5:
            primary_lang = subject_lang_info['language']
            language_confidence = subject_lang_info['confidence']
        else:
            # Default to English if detection confidence is low
            primary_lang = 'en'
            language_confidence = 0
        
        # Translate content if needed and not already in English
        translated_subject = subject
        translated_body = body
        translation_used = False
        
        if self.translate_to_english and primary_lang != 'en' and self.translator:
            # Translate subject
            subject_translation = self.translator.translate(
                subject, source_lang=primary_lang, target_lang='en'
            )
            if subject_translation['success']:
                translated_subject = subject_translation['translated_text']
                translation_used = True
            
            # Translate body (limit size to avoid API limits)
            max_body_length = 5000  # Adjust based on your translation API limits
            truncated_body = body[:max_body_length]
            body_translation = self.translator.translate(
                truncated_body, source_lang=primary_lang, target_lang='en'
            )
            if body_translation['success']:
                translated_body = body_translation['translated_text']
                translation_used = True
        
        # Initialize features dictionary
        features = {
            # Language features
            'primary_language': primary_lang,
            'language_confidence': language_confidence,
            'translation_used': int(translation_used),
            
            # Basic email features
            'subject_length': len(subject),
            'body_length': len(body),
            'reply_to_different': 0,
            
            # URL features
            'url_count': 0,
            'suspicious_tld': 0,
            'ip_url': 0,
            'url_length_avg': 0,
            'domain_age_days': 0,  # Would require external API
            
            # HTML features
            'contains_form': 0,
            'contains_script': 0,
            'contains_iframe': 0,
            
            # Text features
            'suspicious_keywords': 0,
            'exclamation_count': 0,
            'urgency_count': 0,
            'misspelled_domain': 0,
            'security_words_count': 0
        }
        
        # Check for reply-to mismatch
        if reply_to and from_address and reply_to.lower() != from_address.lower():
            features['reply_to_different'] = 1
        
        # Extract and analyze URLs
        urls = self._extract_urls(body)
        features['url_count'] = len(urls)
        
        if urls:
            url_lengths = []
            for url in urls:
                url_lengths.append(len(url))
                
                # Check for IP-based URLs
                if re.search(r'https?://\d+\.\d+\.\d+\.\d+', url):
                    features['ip_url'] = 1
                
                # Check for suspicious TLDs
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                if any(domain.endswith(tld) for tld in self.suspicious_tlds):
                    features['suspicious_tld'] = 1
            
            # Calculate average URL length
            if url_lengths:
                features['url_length_avg'] = sum(url_lengths) / len(url_lengths)
        
        # Check HTML content
        if '<form' in body.lower():
            features['contains_form'] = 1
        if '<script' in body.lower():
            features['contains_script'] = 1
        if '<iframe' in body.lower():
            features['contains_iframe'] = 1
        
        # Count keywords according to language
        # First check in the detected language
        suspicious_count = 0
        if primary_lang in self.suspicious_keywords:
            for keyword in self.suspicious_keywords[primary_lang]:
                if keyword.lower() in body.lower() or keyword.lower() in subject.lower():
                    suspicious_count += 1
        
        # Then check in English if we translated
        if translation_used and 'en' in self.suspicious_keywords:
            for keyword in self.suspicious_keywords['en']:
                if keyword.lower() in translated_body.lower() or keyword.lower() in translated_subject.lower():
                    suspicious_count += 1
        
        features['suspicious_keywords'] = suspicious_count
        
        # Count exclamation marks
        features['exclamation_count'] = body.count('!')
        
        # Count urgency words (language-specific)
        urgency_words = {
            'en': ['urgent', 'immediately', 'now', 'quick', 'hurry', 'fast', 'important'],
            'de': ['dringend', 'sofort', 'jetzt', 'schnell', 'eile', 'wichtig'],
            'fr': ['urgent', 'immédiatement', 'maintenant', 'rapide', 'vite', 'important'],
            'es': ['urgente', 'inmediatamente', 'ahora', 'rápido', 'prisa', 'importante'],
            'it': ['urgente', 'immediatamente', 'ora', 'rapido', 'fretta', 'importante']
        }
        
        urgency_count = 0
        # Check in detected language
        if primary_lang in urgency_words:
            for word in urgency_words[primary_lang]:
                if word in body.lower() or word in subject.lower():
                    urgency_count += 1
        
        # Check in English if translated
        if translation_used:
            for word in urgency_words['en']:
                if word in translated_body.lower() or word in translated_subject.lower():
                    urgency_count += 1
        
        features['urgency_count'] = urgency_count
        
        # Check for misspelled domains
        # Common domains to check for typosquatting
        common_domains = ['paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'bank']
        
        for domain in common_domains:
            # Look for similar domains with one character different or l/1 or o/0 substitutions
            pattern = ''.join([
                f"[{c}|{'.' if i==0 else ''}]" if i == 0 else
                f"[{c}|{'1' if c=='l' else ('0' if c=='o' else '')}]" if c in 'lo' else
                f"[{c}]"
                for i, c in enumerate(domain)
            ])
            if re.search(pattern + r'\.[a-z]+', body.lower()):
                # Check if it's not a legitimate mention of the domain
                if not any(domain in url.lower() for url in urls):
                    features['misspelled_domain'] = 1
                    break
        
        # Count security-related words
        security_words = {
            'en': ['security', 'verify', 'confirm', 'update', 'validate'],
            'de': ['sicherheit', 'bestätigen', 'aktualisieren', 'validieren'],
            'fr': ['sécurité', 'vérifier', 'confirmer', 'mettre à jour', 'valider'],
            'es': ['seguridad', 'verificar', 'confirmar', 'actualizar', 'validar'],
            'it': ['sicurezza', 'verificare', 'confermare', 'aggiornare', 'validare']
        }
        
        security_count = 0
        # Check in detected language
        if primary_lang in security_words:
            for word in security_words[primary_lang]:
                if word in body.lower() or word in subject.lower():
                    security_count += 1
        
        # Check in English if translated
        if translation_used:
            for word in security_words['en']:
                if word in translated_body.lower() or word in translated_subject.lower():
                    security_count += 1
        
        features['security_words_count'] = security_count
        
        # Add untranslated and translated content for reference
        features['original_subject'] = subject
        features['original_body'] = body[:1000]  # Truncate for size
        if translation_used:
            features['translated_subject'] = translated_subject
            features['translated_body'] = translated_body[:1000]  # Truncate for size
        
        return features
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        
        try:
            decoded_header = email.header.decode_header(header)
            header_parts = []
            
            for content, encoding in decoded_header:
                if isinstance(content, bytes):
                    if encoding:
                        try:
                            header_parts.append(content.decode(encoding))
                        except:
                            header_parts.append(content.decode('utf-8', errors='replace'))
                    else:
                        # Try utf-8, fallback to latin-1
                        try:
                            header_parts.append(content.decode('utf-8'))
                        except UnicodeDecodeError:
                            header_parts.append(content.decode('latin-1', errors='replace'))
                else:
                    header_parts.append(content)
            
            return " ".join(header_parts)
        except Exception as e:
            logger.error(f"Error decoding header: {str(e)}")
            return header
    
    def _get_email_body(self, parsed_email) -> str:
        """Extract email body from parsed email"""
        body = ""
        
        try:
            if parsed_email.is_multipart():
                for part in parsed_email.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Get text parts
                    if content_type == "text/plain" or content_type == "text/html":
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                decoded_payload = payload.decode(charset, errors='replace')
                                if content_type == "text/html":
                                    body += self._extract_text_from_html(decoded_payload)
                                else:
                                    body += decoded_payload
                        except Exception as e:
                            logger.error(f"Error processing email part: {str(e)}")
            else:
                # Not multipart - get payload directly
                content_type = parsed_email.get_content_type()
                charset = parsed_email.get_content_charset() or 'utf-8'
                try:
                    payload = parsed_email.get_payload(decode=True)
                    if payload:
                        decoded_payload = payload.decode(charset, errors='replace')
                        if content_type == "text/html":
                            body = self._extract_text_from_html(decoded_payload)
                        else:
                            body = decoded_payload
                except Exception as e:
                    logger.error(f"Error processing email body: {str(e)}")
                    body = str(parsed_email.get_payload())
        except Exception as e:
            logger.error(f"Error extracting email body: {str(e)}")
            # Fallback to raw payload
            body = str(parsed_email.get_payload())
        
        return body
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract readable text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return html_content
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        # Basic URL pattern
        url_pattern = r'https?://[^\s()<>"\\]+'
        
        # Extract and return URLs
        return re.findall(url_pattern, text)

# Example usage
if __name__ == "__main__":
    # Create the feature extractor
    extractor = MultilingualFeatureExtractor()
    
    # Example emails in different languages
    examples = {
        'en': (
            "From: security@paypal.com\n"
            "Subject: Action Required: Verify Your Account\n\n"
            "Dear Customer,\n\n"
            "We have noticed unusual activity on your account. Please verify your "
            "information immediately by clicking the link below:\n\n"
            "https://paypal-security.com/verify.php\n\n"
            "If you do not verify within 24 hours, your account will be suspended.\n\n"
            "PayPal Security Team"
        ),
        'de': (
            "From: sicherheit@paypal.com\n"
            "Subject: Dringend: Bestätigen Sie Ihr Konto\n\n"
            "Sehr geehrter Kunde,\n\n"
            "Wir haben ungewöhnliche Aktivitäten auf Ihrem Konto festgestellt. Bitte bestätigen "
            "Sie Ihre Informationen sofort, indem Sie auf den untenstehenden Link klicken:\n\n"
            "https://paypal-sicherheit.com/bestatigen.php\n\n"
            "Wenn Sie nicht innerhalb von 24 Stunden bestätigen, wird Ihr Konto gesperrt.\n\n"
            "PayPal Sicherheitsteam"
        ),
        'fr': (
            "From: securite@paypal.com\n"
            "Subject: Urgent: Vérifiez votre compte\n\n"
            "Cher client,\n\n"
            "Nous avons remarqué une activité inhabituelle sur votre compte. Veuillez vérifier "
            "vos informations immédiatement en cliquant sur le lien ci-dessous:\n\n"
            "https://paypal-securite.com/verifier.php\n\n"
            "Si vous ne vérifiez pas dans les 24 heures, votre compte sera suspendu.\n\n"
            "Équipe de sécurité PayPal"
        )
    }
    
    # Process each example
    for lang, email_text in examples.items():
        print(f"\nProcessing {lang} email example:")
        features = extractor.extract_features(email_text)
        
        print(f"Detected language: {features['primary_language']} (confidence: {features['language_confidence']:.2f})")
        print(f"Translation used: {'Yes' if features['translation_used'] == 1 else 'No'}")
        print(f"Suspicious keywords: {features['suspicious_keywords']}")
        print(f"Urgency words: {features['urgency_count']}")
        print(f"Security words: {features['security_words_count']}")
        print(f"Contains form: {'Yes' if features['contains_form'] else 'No'}")
        print(f"Suspicious URLs: {features['url_count']} URLs, suspicious TLD: {'Yes' if features['suspicious_tld'] else 'No'}")
        print("-" * 50)
