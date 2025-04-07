# Deep Learning Models for Phishing Detection

이 디렉토리는 이메일 피싱 탐지를 위한 딥러닝 모델들을 포함하고 있습니다.

## 구현된 모델

### BERT 기반 피싱 분류기 (bert_model.py)

BERT(Bidirectional Encoder Representations from Transformers)를 기반으로 한 피싱 이메일 분류 모델입니다.

#### 주요 특징
- 사전 훈련된 BERT 모델을 활용하여 이메일 텍스트 이해
- 문맥 기반 텍스트 분석으로 높은 정확도 제공
- 피싱 이메일의 미묘한 특성까지 포착 가능

#### 요구 사항
- Python 3.8+
- PyTorch 1.7+
- Transformers 4.0+
- CUDA 지원 GPU (권장)

#### 설치
```bash
pip install torch transformers
```

#### 사용 방법
```python
from ml.deep_learning.bert_model import BertPhishingDetector

# 모델 초기화
detector = BertPhishingDetector()

# 이메일로 예측
email_text = "Subject: Urgent action required..." 
result = detector.predict(email_text)

print(f"Phishing probability: {result['probabilities']['phishing']}")
```

#### 모델 훈련

```python
# 준비된 훈련 데이터
train_texts = [...]  # 이메일 텍스트 리스트
train_labels = [...]  # 레이블 (0: 정상, 1: 피싱)

# 모델 훈련
detector.train(train_texts, train_labels, epochs=4, batch_size=16)

# 모델 저장
detector.save_model('models/bert_phishing_detector.pth')
```

## 계획된 모델

### LSTM 기반 시퀀스 분석기 (Coming Soon)
Long Short-Term Memory 네트워크를 사용하여 이메일 시퀀스 데이터의 패턴을 학습합니다.

### CNN 기반 HTML 분석기 (Coming Soon)
이메일 HTML 구조를 분석하기 위한 Convolutional Neural Network 모델입니다.

## 통합 가이드

기존 시스템에 딥러닝 모델을 통합하려면 다음 단계를 따라주세요:

1. 필요한 의존성 설치:
```bash
pip install -r ml/deep_learning/requirements.txt
```

2. `app.py`에 모델 임포트 추가:
```python
from ml.deep_learning.bert_model import BertPhishingDetector, prepare_email_for_bert
```

3. 모델 초기화 코드 추가:
```python
# BERT 모델 초기화 (기존 RandomForest 모델과 함께 사용)
bert_model_path = os.path.join(app.config['MODEL_FOLDER'], 'bert_phishing_model.pth')
try:
    bert_detector = BertPhishingDetector(model_path=bert_model_path)
except Exception as e:
    print(f"BERT 모델을 로드하는데 실패했습니다: {e}")
    bert_detector = None
```

4. 예측 코드에 BERT 모델 통합:
```python
# 기존 모델 예측
features = extract_features_from_email(email_text)
prediction = model.predict([features])
proba = model.predict_proba([features])

# BERT 모델 예측 (가능한 경우)
if bert_detector:
    bert_ready_text = prepare_email_for_bert(email_text)
    bert_result = bert_detector.predict(bert_ready_text)
    
    # 두 모델의 결과 조합
    combined_phishing_prob = (proba[0][1] + bert_result['probabilities']['phishing']) / 2
    combined_prediction = 1 if combined_phishing_prob > 0.5 else 0
else:
    combined_prediction = prediction[0]
    combined_phishing_prob = proba[0][1] if prediction[0] == 1 else (1 - proba[0][0])
```

## 성능 고려 사항

- BERT 모델은 CPU보다 GPU에서 훨씬 빠르게 동작합니다.
- 대용량 이메일 처리 시 배치 처리를 고려하세요.
- 최초 모델 로딩에 시간이 걸릴 수 있습니다.