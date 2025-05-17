import pytest

from ml.multilingual.language_detector import LanguageDetector


def test_english_detection_with_ngrams():
    text = "This is a simple English sentence used for testing the detection mechanism."
    detector = LanguageDetector()
    result = detector.detect_language(text)
    assert result['language'] == 'en'
    assert result['confidence'] > 0
    assert result['method'] == 'ngram'


def test_insufficient_text():
    detector = LanguageDetector()
    result = detector.detect_language("Hi")
    assert result['language'] == 'unknown'
    assert result['method'] == 'insufficient_text'
