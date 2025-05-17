import importlib
from ml.multilingual import language_detector


def create_detector_without_external(monkeypatch):
    monkeypatch.setattr(language_detector, "LANGID_AVAILABLE", False)
    monkeypatch.setattr(language_detector, "SPACY_AVAILABLE", False)
    return language_detector.LanguageDetector()


def test_ngram_detection_english(monkeypatch):
    detector = create_detector_without_external(monkeypatch)
    text = "This is a simple English text for testing the language detector."
    result = detector.detect_language(text)
    assert result["language"] == "en"
    assert result["confidence"] > 0
    assert result["method"] == "ngram"


def test_insufficient_text(monkeypatch):
    detector = create_detector_without_external(monkeypatch)
    text = "Hi"
    result = detector.detect_language(text)
    assert result == {
        "language": "unknown",
        "confidence": 0.0,
        "method": "insufficient_text",
    }

