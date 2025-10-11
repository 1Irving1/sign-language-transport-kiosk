# 수어 대중교통 키오스크

농인을 위한 수어 기반 대중교통 플랫폼

## 프로젝트 개요

이 프로젝트는 농인 사용자들이 수어(수화)를 통해 기차 예매를 할 수 있는 키오스크 시스템입니다. AI 모델을 활용하여 수어를 인식하고, 실시간으로 대중교통 정보를 제공합니다.

## 주요 기능

- **수어 인식**: 카메라를 통한 실시간 수어 인식
- **기차 예매**: 수어로 목적지, 시간, 인원, 좌석 등 입력
- **실시간 정보**: 코레일 API 연동을 통한 실시간 열차 정보
- **결제 시스템**: 수어로 결제 방법 선택 및 처리
- **QR코드 생성**: 예매 완료 후 QR코드 생성
- **카카오톡 연동**: 예매 정보 카카오톡으로 전송

## 기술 스택

### 백엔드
- Java 17
- Spring Boot 3.2.5
- Spring Data JPA
- Spring WebFlux
- Spring Security
- H2/MySQL Database
- ZXing (QR코드 생성)

### AI 모델
- MediaPipe (키포인트 추출)
- GRU 기반 수어 인식 모델
- 실시간 추론 시스템

## 프로젝트 구조

```
Capstone/
├── backend/                    # Spring Boot 백엔드
│   ├── src/main/java/
│   │   └── com/capstone/
│   │       ├── controller/     # REST API 컨트롤러
│   │       ├── service/        # 비즈니스 로직
│   │       ├── entity/         # JPA 엔티티
│   │       ├── dto/           # 데이터 전송 객체
│   │       ├── repository/    # 데이터 접근 계층
│   │       └── util/         # 유틸리티
│   └── src/main/resources/
│       ├── application.yml
│       ├── application-dev.yml
│       └── application-prod.yml
└── Data/                      # AI 모델 및 데이터
    └── 03.AI모델 2/
        └── NIA_CSLR/         # 수어 인식 모델
```

## 개발 환경 설정

### 백엔드 실행
```bash
cd backend
mvn clean install
mvn spring-boot:run
```

### API 엔드포인트
- Base URL: `http://localhost:8080/api`
- H2 콘솔: `http://localhost:8080/h2-console`

## 주요 API

### 기차 조회
- `POST /api/train/search` - 슬롯 데이터로 기차 검색
- `GET /api/train/search` - 파라미터로 기차 검색

### 예매
- `POST /api/booking` - 예매 생성
- `GET /api/booking/{bookingNumber}` - 예매 조회

### 결제
- `POST /api/payment/{bookingId}` - 결제 처리
- `GET /api/payment/{paymentId}` - 결제 조회

## 데이터 흐름

1. **수어 입력**: 사용자가 카메라 앞에서 수어로 정보 입력
2. **AI 인식**: MediaPipe로 키포인트 추출 → GRU 모델로 수어 인식
3. **슬롯 변환**: 인식 결과를 JSON 슬롯으로 변환
4. **API 연동**: 코레일 API로 실시간 열차 정보 조회
5. **결과 제공**: 추천 노선을 카드 UI로 표시
6. **예매/결제**: QR코드 생성, 카카오톡 전송

## 향후 개발 계획

- [ ] 코레일 API 실제 연동
- [ ] 카카오톡 API 연동
- [ ] 지하철 API 연동
- [ ] 실제 결제 시스템 연동
- [ ] 보안 강화 (JWT 토큰, 암호화)
- [ ] 모니터링 및 로깅

## 라이선스

이 프로젝트는 교육 목적으로 개발되었습니다.

## 기여자

- 백엔드 개발자
- AI 모델 개발자
- 프론트엔드 개발자

