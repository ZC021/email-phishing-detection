import re
import string
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('language_detector')

class LanguageDetector:
    """
    Detects the language of text using various methods.
    Primary method is langid, with fallbacks to spaCy and n-gram analysis.
    """
    
    def __init__(self):
        self.langid_model = None
        self.spacy_models = {}
        
        # Initialize langid if available
        if LANGID_AVAILABLE:
            import langid
            self.langid_model = langid
            logger.info("Initialized langid language detection")
        
        # Initialize basic spaCy models if available
        if SPACY_AVAILABLE:
            try:
                # Try to load common language models if installed
                for lang_code in ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ja', 'zh']:
                    try:
                        model_name = self._get_spacy_model_name(lang_code)
                        if spacy.util.is_package(model_name):
                            self.spacy_models[lang_code] = spacy.load(model_name)
                            logger.info(f"Loaded spaCy model for {lang_code}")
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Error loading spaCy models: {str(e)}")
    
    def _get_spacy_model_name(self, lang_code: str) -> str:
        """Get the appropriate spaCy model name for a language code"""
        # Map of language codes to spaCy model names
        model_map = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ja': 'ja_core_news_sm',
            'zh': 'zh_core_web_sm'
        }
        
        return model_map.get(lang_code, f"{lang_code}_core_news_sm")
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.
        Returns a dictionary with language code, confidence, and method used.
        """
        if not text or len(text.strip()) < 10:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_text'}
        
        # Try langid first (most reliable)
        if self.langid_model:
            try:
                lang_code, confidence = self.langid_model.classify(text)
                return {
                    'language': lang_code,
                    'confidence': confidence,
                    'method': 'langid'
                }
            except Exception as e:
                logger.warning(f"langid detection failed: {str(e)}")
        
        # Try spaCy if available and langid failed
        if self.spacy_models:
            try:
                best_lang = None
                best_score = -1
                
                # Use a sample of text to speed up processing
                sample = text[:500]
                
                for lang_code, nlp in self.spacy_models.items():
                    doc = nlp(sample)
                    
                    # Simple scoring based on token recognition
                    recognized_tokens = sum(1 for token in doc if not token.is_oov)
                    total_tokens = len(doc)
                    
                    if total_tokens > 0:
                        score = recognized_tokens / total_tokens
                        if score > best_score:
                            best_score = score
                            best_lang = lang_code
                
                if best_lang and best_score > 0.3:  # Reasonable threshold
                    return {
                        'language': best_lang,
                        'confidence': best_score,
                        'method': 'spacy'
                    }
            except Exception as e:
                logger.warning(f"spaCy detection failed: {str(e)}")
        
        # Fallback to simple n-gram analysis
        return self._detect_with_ngrams(text)
    
    def _detect_with_ngrams(self, text: str) -> Dict[str, Any]:
        """
        Fallback method using character n-gram frequency analysis.
        Very basic but works as a last resort.
        """
        # Language n-gram frequency profiles (simplified)
        lang_profiles = {
            'en': {'th': 0.027, 'he': 0.025, 'in': 0.023, 'er': 0.021, 'an': 0.021, 
                   'on': 0.018, 're': 0.016, 'ed': 0.015, 'nd': 0.015, 'ha': 0.014},
            'es': {'de': 0.031, 'en': 0.024, 'el': 0.021, 'la': 0.021, 'ar': 0.019,
                   'es': 0.019, 'os': 0.018, 'ue': 0.015, 'ra': 0.015, 'qu': 0.014},
            'fr': {'es': 0.028, 'le': 0.023, 'de': 0.022, 'en': 0.022, 'on': 0.021,
                   'nt': 0.018, 'la': 0.018, 'ou': 0.017, 're': 0.015, 'an': 0.015},
            'de': {'en': 0.035, 'er': 0.030, 'ch': 0.023, 'in': 0.022, 'de': 0.019,
                  'nd': 0.018, 'ie': 0.017, 'ge': 0.015, 'ei': 0.014, 'te': 0.014}
            # Add more language profiles as needed
        }
        
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        # Extract bigrams
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = {}
        
        for bigram in bigrams:
            if bigram not in bigram_counts:
                bigram_counts[bigram] = 0
            bigram_counts[bigram] += 1
        
        # Convert to frequencies
        total_bigrams = sum(bigram_counts.values())
        if total_bigrams == 0:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'ngram_fail'}
            
        frequencies = {bigram: count/total_bigrams for bigram, count in bigram_counts.items()}
        
        # Compare to language profiles
        best_lang = 'unknown'
        best_score = 0
        
        for lang, profile in lang_profiles.items():
            score = 0
            for bigram, expected_freq in profile.items():
                if bigram in frequencies:
                    # Calculate how close the frequency is to expected
                    similarity = 1 - abs(frequencies[bigram] - expected_freq) / expected_freq
                    score += similarity * expected_freq  # Weight by importance of bigram
            
            if score > best_score:
                best_score = score
                best_lang = lang
        
        # Normalize confidence score to 0-1 range
        confidence = min(best_score / 0.5, 1.0)
        
        return {
            'language': best_lang,
            'confidence': confidence,
            'method': 'ngram'
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of languages with full support"""
        supported = []
        
        if LANGID_AVAILABLE:
            # langid supports many languages, but we list common ones
            supported = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                         'ko', 'ar', 'hi', 'bn', 'pa', 'te', 'mr', 'ta', 'ur', 'fa',
                         'sv', 'no', 'da', 'fi', 'hu', 'pl', 'cs', 'ro', 'bg', 'el',
                         'tr', 'th', 'vi']
        else:
            # Fall back to languages with spaCy models
            supported = list(self.spacy_models.keys())
            
            # Fall back to languages with n-gram profiles
            if not supported:
                supported = ['en', 'es', 'fr', 'de']
        
        return supported

# Example usage
if __name__ == "__main__":
    detector = LanguageDetector()
    
    # Test with different languages
    texts = {
        "This is a sample text in English language.": "en",
        "Dies ist ein Beispieltext in deutscher Sprache.": "de",
        "Ceci est un exemple de texte en langue française.": "fr",
        "Este es un texto de ejemplo en idioma español.": "es"
    }
    
    for text, expected in texts.items():
        result = detector.detect_language(text)
        print(f"Text: {text[:30]}...")
        print(f"Detected: {result['language']} (expected: {expected})")
        print(f"Confidence: {result['confidence']:.4f}, Method: {result['method']}")
        print("-" * 50)
    
    print(f"Supported languages: {detector.get_supported_languages()}")
