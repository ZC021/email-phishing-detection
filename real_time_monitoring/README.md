# 실시간 이메일 모니터링 (Real-time Email Monitoring)

이 모듈은 IMAP 프로토콜을 통해 이메일 계정에 연결하여 실시간으로 수신되는 이메일을 모니터링하고 피싱 공격을 감지합니다.

## 주요 기능

- 다중 이메일 계정 모니터링 동시 지원
- 메이저 이메일 공급자(Gmail, Outlook, Yahoo 등) 자동 감지
- 비동기 및 멀티스레드 이메일 처리
- 머신러닝 및 딥러닝 모델과의 통합
- 피싱 이메일 감지 시 알림 시스템

## 아키텍처

시스템은 다음과 같은 구성 요소로 이루어져 있습니다:

1. **EmailMonitor**: 이메일 계정에 연결하고 새 이메일을 모니터링하는 코어 클래스
2. **MonitoringService**: EmailMonitor와 피싱 탐지 모델을 통합하는 서비스 계층
3. **Flask 통합**: 웹 애플리케이션과의 통합을 위한 유틸리티 함수

## 사용법

### 독립 실행형 모니터링 서비스

```python
from real_time_monitoring.integration import MonitoringService

# 알림 콜백 함수 정의
def notification_handler(detection_result):
    print(f"피싱 이메일 감지됨: {detection_result['email_info']['subject']}")
    print(f"신뢰도: {detection_result['confidence'] * 100:.2f}%")

# 서비스 초기화
service = MonitoringService(
    config_path="config/monitoring.json",
    model_path="models/phishing_model.joblib",
    bert_model_path="models/bert_phishing_model.pth",
    notification_callback=notification_handler
)

# 이메일 계정 추가
service.add_account(
    email="your-email@example.com",
    password="your-secure-password",
    check_interval=60  # 60초마다 확인
)

# 모니터링 시작
service.start()

# ... 애플리케이션 로직 ...

# 모니터링 중지
service.stop()
```

### Flask 애플리케이션과 통합

```python
from flask import Flask
from real_time_monitoring.integration import setup_monitoring_service, start_monitoring

app = Flask(__name__)
app.config['MODEL_FOLDER'] = 'models'
app.config['CONFIG_FOLDER'] = 'config'

# 모니터링 서비스 설정
monitoring_service = setup_monitoring_service(app)

# 라우트에서 모니터링 서비스 사용
@app.route('/monitoring/start')
def start_monitoring_route():
    success = start_monitoring(app)
    return {"success": success}

@app.route('/monitoring/detections')
def get_detections():
    if hasattr(app, 'monitoring_service'):
        detections = app.monitoring_service.get_detected_phishing()
        return {"detections": detections}
    return {"detections": []}

if __name__ == '__main__':
    app.run(debug=True)
```

## 보안 고려 사항

### 이메일 자격 증명 보호

이 모듈은 사용자의 이메일 계정에 접근하므로 보안이 매우 중요합니다:

1. **환경 변수 사용**: 코드에 직접 자격 증명을 하드코딩하지 마세요.
2. **암호화된 설정 파일**: 구성 파일을 암호화하여 저장하세요.
3. **앱 비밀번호 사용**: Gmail 같은 서비스의 경우 계정 비밀번호 대신 앱 비밀번호를 사용하세요.

### 예시:

```python
import os
from real_time_monitoring.integration import MonitoringService

service = MonitoringService()

# 환경 변수에서 자격 증명 가져오기
service.add_account(
    email=os.environ.get('EMAIL_USERNAME'),
    password=os.environ.get('EMAIL_PASSWORD')
)
```

## 설치 요구 사항

```
pip install imaplib email threading queue
```

## 설정 파일 구조

모니터링 서비스는 JSON 형식의 설정 파일을 사용합니다:

```json
{
  "accounts": [
    {
      "email": "user@example.com",
      "password": "********",  // 실제 배포 시 암호화 권장
      "imap_server": "imap.example.com",
      "imap_port": 993,
      "use_ssl": true,
      "check_interval": 60
    }
  ],
  "detection": {
    "model_path": "models/phishing_model.joblib",
    "bert_model_path": "models/bert_phishing_model.pth",
    "confidence_threshold": 0.75
  },
  "notification": {
    "enabled": true,
    "email_notification": true,
    "web_notification": true,
    "admin_email": "admin@example.com"
  }
}
```

## 한계 및 향후 개선 사항

1. **이메일 처리 성능**: 대용량 메일함의 경우 초기 처리에 시간이 소요될 수 있습니다.
2. **자격 증명 저장**: 현재는 기본적인 보안만 제공하며, 더 강력한 암호화가 필요합니다.
3. **폴링 방식의 한계**: IMAP 프로토콜의 특성상 폴링 방식을 사용하며, 푸시 알림 방식이 더 효율적일 수 있습니다.

향후 개선 예정:
- 더 효율적인 푸시 알림 기반 모니터링
- OAuth 인증 지원
- 위험 수준별 필터링 및 알림 설정
- 이메일 클라이언트 플러그인 연동

## 기여 방법

이 모듈에 기여하고 싶다면 다음과 같은 영역에서 도움을 줄 수 있습니다:

1. 다양한 이메일 서비스 지원 확대
2. 보안 강화 (자격 증명 암호화, OAuth 지원)
3. 성능 최적화
4. 테스트 케이스 추가

## 라이센스
MIT 라이센스에 따라 배포됩니다.
