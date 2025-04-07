# 다국어 피싱 탐지 모듈 (Multilingual Phishing Detection)

이 모듈은 여러 언어로 작성된 이메일에서 피싱을 탐지하기 위한 기능을 제공합니다. 자동 언어 감지, 번역, 그리고 언어별 맞춤형 특성 추출을 지원합니다.

## 주요 기능

- **자동 언어 감지**: 이메일 제목과 본문에서 사용된 언어를 자동으로 감지
- **다국어 특성 추출**: 각 언어에 맞는 피싱 특성을 추출
- **번역 통합**: 필요시 텍스트를 영어로 번역하여 기존 모델 활용
- **지원 언어 확장**: 기본적으로 영어, 독일어, 프랑스어, 스페인어, 이탈리아어를 지원하며 쉽게 확장 가능

## 구성 요소

1. **LanguageDetector**: 텍스트의 언어를 감지하는 클래스
2. **Translator**: 필요시 이메일 내용을 영어로 번역하는 클래스
3. **MultilingualFeatureExtractor**: 다양한 언어의 이메일에서 피싱 탐지 특성을 추출하는 클래스

## 설치 요구 사항

```bash
# 기본 의존성
pip install langid spacy beautifulsoup4 requests

# 다국어 처리를 위한 spaCy 모델 설치
python -m spacy download en_core_web_sm  # 영어
python -m spacy download de_core_news_sm  # 독일어
python -m spacy download fr_core_news_sm  # 프랑스어
python -m spacy download es_core_news_sm  # 스페인어
python -m spacy download it_core_news_sm  # 이탈리아어

# 선택적 의존성 (번역에 필요)
pip install translatepy
```

## 사용 방법

### 기본 사용법

```python
from ml.multilingual.feature_extractor import MultilingualFeatureExtractor

# 특성 추출기 초기화
extractor = MultilingualFeatureExtractor(
    suspicious_keywords_path="config/multilingual_keywords.json",
    translate_to_english=True  # 영어로 번역할지 여부
)

# 이메일에서 특성 추출
email_text = "From: security@paypal.com\nSubject: Urgent: Confirm your account\n\n..."
features = extractor.extract_features(email_text)

# 감지된 언어 및 주요 특성 확인
print(f"Detected language: {features['primary_language']}")
print(f"Suspicious keywords: {features['suspicious_keywords']}")
print(f"URLs found: {features['url_count']}")
```

### 번역 API 설정

번역 기능을 사용하려면 환경 변수나 코드에서 API 키를 설정해야 합니다:

```python
import os
from ml.multilingual.translator import Translator

# 환경 변수로 API 키 설정 (권장)
os.environ['GOOGLE_TRANSLATE_API_KEY'] = 'your-api-key'
os.environ['AZURE_TRANSLATE_API_KEY'] = 'your-api-key'
os.environ['DEEPL_TRANSLATE_API_KEY'] = 'your-api-key'

# 또는 코드에서 직접 설정
translator = Translator(api_keys={
    'google': 'your-google-api-key',
    'azure': 'your-azure-api-key',
    'deepl': 'your-deepl-api-key'
})
```

### 커스텀 키워드 추가

JSON 파일을 통해 언어별 의심스러운 키워드를 추가할 수 있습니다:

```json
{
  "en": [
    "verify immediately", "unusual activity", "security breach"
  ],
  "de": [
    "sofort bestätigen", "ungewöhnliche Aktivität", "Sicherheitsverstoß"
  ],
  "fr": [
    "vérifier immédiatement", "activité inhabituelle", "violation de sécurité"
  ],
  "ko": [
    "즉시 확인", "비정상적인 활동", "보안 위반"
  ]
}
```

## 기존 시스템과의 통합

### 기존 모델에 다국어 지원 추가

```python
from ml.feature_extraction import extract_features_from_email
from ml.multilingual.feature_extractor import MultilingualFeatureExtractor
from ml.model import PhishingDetectionModel

# 기존 및 다국어 특성 추출기 초기화
standard_model = PhishingDetectionModel()
multilingual_extractor = MultilingualFeatureExtractor()

def enhanced_extract_features(email_text):
    # 언어 감지 및 다국어 특성 추출
    multi_features = multilingual_extractor.extract_features(email_text)
    
    # 기존 특성 추출 (영어만 지원)
    standard_features = extract_features_from_email(email_text)
    
    # 언어가 영어가 아닌 경우 다국어 특성 활용
    if multi_features['primary_language'] != 'en' and multi_features['translation_used'] == 1:
        # 번역된 텍스트를 기존 추출기에 제공
        translated_email = f"From: {email_text.get('from', '')}\nSubject: {multi_features['translated_subject']}\n\n{multi_features['translated_body']}"
        standard_features = extract_features_from_email(translated_email)
    
    # 다국어 특성 추가
    standard_features.update({
        'language': multi_features['primary_language'],
        'translation_used': multi_features['translation_used'],
        'multilingual_suspicious_keywords': multi_features['suspicious_keywords']
    })
    
    return standard_features
```

## 언어별 피싱 패턴

각 언어마다 피싱 이메일에 나타나는 특징적인 패턴이 있습니다:

| 언어 | 주요 패턴 | 일반적인 위장 대상 |
|------|----------|------------|
| 영어 | "Verify immediately", "Account suspended" | PayPal, Apple, Microsoft |
| 독일어 | "Konto bestätigen", "Sicherheitshinweis" | Deutsche Bank, Amazon.de |
| 프랑스어 | "Vérifiez votre compte", "Alerte de sécurité" | Crédit Agricole, Orange |
| 스페인어 | "Verificar cuenta", "Alerta de seguridad" | Santander, BBVA |
| 이탈리아어 | "Verifica il tuo conto", "Avviso di sicurezza" | UniCredit, Poste Italiane |

## 성능 고려 사항

다국어 지원을 추가할 때 다음 사항을 고려해야 합니다:

1. **언어 감지 정확도**: 짧은 텍스트에서는 언어 감지가 부정확할 수 있으며, 여러 언어가 혼합된 이메일은 탐지가 어려울 수 있습니다.
2. **번역 API 제한**: 외부 번역 API를 사용할 경우 요청 제한과 비용을 고려해야 합니다.
3. **처리 시간**: 언어 감지와 번역은 추가 처리 시간이 필요합니다.

## 향후 개선 방향

1. **더 많은 언어 지원**: 아시아 언어(한국어, 일본어, 중국어 등) 및 다른 유럽 언어 지원 확대
2. **언어별 모델**: 각 언어에 특화된 피싱 탐지 모델 개발
3. **효율적인 번역**: 전체 이메일이 아닌 중요 부분만 선택적으로 번역하여 효율성 개선
4. **다국어 단어 임베딩**: 언어 간 의미를 보존하는 다국어 단어 임베딩 활용

## 기여 방법

다국어 지원을 개선하고 싶다면 다음과 같은 방법으로 기여할 수 있습니다:

1. 새로운 언어에 대한 의심스러운 키워드 목록 추가
2. 언어별 피싱 패턴 분석 및 문서화
3. 다국어 데이터셋 구축 및 공유
4. 번역 없이도 다국어 이메일을 처리할 수 있는 모델 개발

## 데이터셋 및 참고 자료

- [Phishing Email Corpus (다국어)](https://github.com/openphish/phishing-dataset)
- [Anti-Phishing Working Group (APWG) 보고서](https://apwg.org/trendsreports/)
- [다국어 NLP 라이브러리 및 리소스](https://github.com/sebastianruder/NLP-progress/blob/master/multilingual.md)
