import json
import requests
import logging
import os
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('translator')

class Translator:
    """
    Translates text between languages using different translation services.
    Supports multiple translation backends with fallback mechanisms.
    """
    
    def __init__(self, default_target_lang: str = 'en', api_keys: Dict[str, str] = None):
        self.default_target_lang = default_target_lang
        self.api_keys = api_keys or {}
        
        # Try to load API keys from environment if not provided
        for service in ['google', 'azure', 'deepl']:
            env_var = f"{service.upper()}_TRANSLATE_API_KEY"
            if service not in self.api_keys and env_var in os.environ:
                self.api_keys[service] = os.environ[env_var]
        
        # Available translation services
        self.available_services = []
        
        # Check which services are available based on API keys
        if 'google' in self.api_keys:
            self.available_services.append('google')
        
        if 'azure' in self.api_keys:
            self.available_services.append('azure')
            
        if 'deepl' in self.api_keys:
            self.available_services.append('deepl')
            
        # Setup a fallback built-in translator for emergency
        try:
            import translatepy
            self.translatepy_available = True
            self.available_services.append('translatepy')
        except ImportError:
            self.translatepy_available = False
        
        if not self.available_services:
            logger.warning("No translation services available. Translation will be limited.")
    
    def translate(self, text: str, source_lang: str = None, target_lang: str = None,
                 preferred_service: str = None) -> Dict[str, Any]:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code (defaults to self.default_target_lang)
            preferred_service: Preferred translation service to use
            
        Returns:
            Dictionary containing translated text, detected source language,
            target language, and service used
        """
        if not text or len(text.strip()) == 0:
            return {
                'translated_text': '',
                'source_lang': source_lang or 'unknown',
                'target_lang': target_lang or self.default_target_lang,
                'service': 'none',
                'success': False,
                'error': 'Empty text provided'
            }
        
        # Set default target language if not specified
        if target_lang is None:
            target_lang = self.default_target_lang
        
        # If the text is already in the target language, return as is
        if source_lang == target_lang and source_lang is not None:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'service': 'none',
                'success': True,
                'error': None
            }
        
        # Determine which translation services to try, in order
        services_to_try = []
        
        if preferred_service and preferred_service in self.available_services:
            services_to_try.append(preferred_service)
        
        # Add remaining available services
        for service in self.available_services:
            if service not in services_to_try:
                services_to_try.append(service)
        
        # Try each translation service until successful
        last_error = None
        for service in services_to_try:
            try:
                if service == 'google':
                    result = self._translate_google(text, source_lang, target_lang)
                elif service == 'azure':
                    result = self._translate_azure(text, source_lang, target_lang)
                elif service == 'deepl':
                    result = self._translate_deepl(text, source_lang, target_lang)
                elif service == 'translatepy' and self.translatepy_available:
                    result = self._translate_local(text, source_lang, target_lang)
                else:
                    continue
                
                if result['success']:
                    return result
            except Exception as e:
                logger.warning(f"Translation with {service} failed: {str(e)}")
                last_error = str(e)
        
        # If all services failed, return error
        return {
            'translated_text': text,  # Return original text
            'source_lang': source_lang or 'unknown',
            'target_lang': target_lang,
            'service': 'none',
            'success': False,
            'error': last_error or 'All translation services failed'
        }
    
    def _translate_google(self, text: str, source_lang: Optional[str], target_lang: str) -> Dict[str, Any]:
        """
        Translate text using Google Cloud Translation API
        """
        api_key = self.api_keys.get('google')
        if not api_key:
            raise ValueError("Google Translate API key not provided")
        
        url = "https://translation.googleapis.com/language/translate/v2"
        payload = {
            'q': text,
            'target': target_lang,
            'key': api_key,
            'format': 'text'
        }
        
        if source_lang:
            payload['source'] = source_lang
        
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            translation = result['data']['translations'][0]
            
            return {
                'translated_text': translation['translatedText'],
                'source_lang': translation.get('detectedSourceLanguage', source_lang or 'unknown'),
                'target_lang': target_lang,
                'service': 'google',
                'success': True,
                'error': None
            }
        else:
            raise Exception(f"Google Translate API error: {response.text}")
    
    def _translate_azure(self, text: str, source_lang: Optional[str], target_lang: str) -> Dict[str, Any]:
        """
        Translate text using Azure Translator API
        """
        api_key = self.api_keys.get('azure')
        if not api_key:
            raise ValueError("Azure Translator API key not provided")
        
        endpoint = "https://api.cognitive.microsofttranslator.com/translate"
        location = os.environ.get('AZURE_TRANSLATOR_LOCATION', 'global')
        
        params = {
            'api-version': '3.0',
            'to': target_lang
        }
        
        if source_lang:
            params['from'] = source_lang
        
        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json'
        }
        
        body = [{
            'text': text
        }]
        
        response = requests.post(endpoint, params=params, headers=headers, json=body)
        
        if response.status_code == 200:
            result = response.json()
            translation = result[0]['translations'][0]
            
            detected_lang = None
            if 'detectedLanguage' in result[0]:
                detected_lang = result[0]['detectedLanguage']['language']
            
            return {
                'translated_text': translation['text'],
                'source_lang': detected_lang or source_lang or 'unknown',
                'target_lang': target_lang,
                'service': 'azure',
                'success': True,
                'error': None
            }
        else:
            raise Exception(f"Azure Translator API error: {response.text}")
    
    def _translate_deepl(self, text: str, source_lang: Optional[str], target_lang: str) -> Dict[str, Any]:
        """
        Translate text using DeepL API
        """
        api_key = self.api_keys.get('deepl')
        if not api_key:
            raise ValueError("DeepL API key not provided")
        
        url = "https://api-free.deepl.com/v2/translate"
        
        # Convert language codes to DeepL format
        deepl_target = self._convert_to_deepl_lang(target_lang)
        
        payload = {
            'text': text,
            'target_lang': deepl_target,
            'auth_key': api_key
        }
        
        if source_lang:
            deepl_source = self._convert_to_deepl_lang(source_lang)
            payload['source_lang'] = deepl_source
        
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            translation = result['translations'][0]
            
            return {
                'translated_text': translation['text'],
                'source_lang': translation.get('detected_source_language', source_lang or 'unknown').lower(),
                'target_lang': target_lang,
                'service': 'deepl',
                'success': True,
                'error': None
            }
        else:
            raise Exception(f"DeepL API error: {response.text}")
    
    def _translate_local(self, text: str, source_lang: Optional[str], target_lang: str) -> Dict[str, Any]:
        """
        Translate text using local translatepy library (fallback)
        """
        try:
            from translatepy import Translator as TPYTranslator
            translator = TPYTranslator()
            
            # Convert 2-letter codes to language names if needed
            translation = translator.translate(text, source_lang, target_lang)
            
            return {
                'translated_text': translation.result,
                'source_lang': translation.source,
                'target_lang': translation.destination,
                'service': 'translatepy',
                'success': True,
                'error': None
            }
        except Exception as e:
            raise Exception(f"Local translation error: {str(e)}")
    
    def _convert_to_deepl_lang(self, lang_code: str) -> str:
        """
        Convert ISO language code to DeepL format
        """
        # DeepL uses uppercase language codes
        lang_code = lang_code.upper()
        
        # Special case handling
        if lang_code == 'EN':
            return 'EN-US'  # Default to US English
        elif lang_code == 'PT':
            return 'PT-PT'  # European Portuguese
        
        return lang_code

# Example usage
if __name__ == "__main__":
    # Create translator with fallback options
    translator = Translator()
    
    # Test with different languages
    texts = {
        "This is a test message in English.": "en",
        "Dies ist eine Testnachricht auf Deutsch.": "de",
        "Ceci est un message de test en français.": "fr",
        "Este es un mensaje de prueba en español.": "es"
    }
    
    for text, source_lang in texts.items():
        print(f"Original ({source_lang}): {text}")
        
        # Translate to English
        result = translator.translate(text, source_lang=source_lang, target_lang='en')
        if result['success']:
            print(f"Translated (en): {result['translated_text']}")
            print(f"Service used: {result['service']}")
        else:
            print(f"Translation failed: {result['error']}")
        
        print("-" * 50)
