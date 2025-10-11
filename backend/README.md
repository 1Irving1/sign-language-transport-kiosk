# Sign Language Transport Backend

농인을 위한 수어 기반 대중교통 플랫폼 백엔드 서비스

## 기술 스택

- **Java 17**
- **Spring Boot 3.2.5**
- **Spring Data JPA**
- **Spring WebFlux** (비동기 HTTP 클라이언트)
- **Spring Security**
- **H2 Database** (개발용) / **MySQL** (운영용)
- **ZXing** (QR코드 생성)
- **Jackson** (JSON 처리)

## 프로젝트 구조

```
src/main/java/com/capstone/
├── SignLanguageTransportApplication.java    # 메인 애플리케이션
├── config/                                 # 설정 클래스
│   ├── WebConfig.java
│   └── SecurityConfig.java
├── controller/                             # REST API 컨트롤러
│   ├── TrainController.java
│   ├── BookingController.java
│   └── PaymentController.java
├── service/                                # 비즈니스 로직
│   ├── KorailService.java
│   ├── BookingService.java
│   └── PaymentService.java
├── entity/                                 # JPA 엔티티
│   ├── Booking.java
│   └── Payment.java
├── dto/                                     # 데이터 전송 객체
│   ├── SlotDataDto.java
│   ├── TrainInfoDto.java
│   └── BookingRequestDto.java
├── repository/                              # 데이터 접근 계층
│   ├── BookingRepository.java
│   └── PaymentRepository.java
└── util/                                    # 유틸리티
    ├── JsonParser.java
    └── QrGenerator.java
```

## 주요 기능

### 1. 기차 조회 (Train)
- AI에서 받은 슬롯 데이터로 기차 검색
- 코레일 API 연동 (현재 Mock 데이터)
- 실시간 열차 정보 조회

### 2. 예매 (Booking)
- 기차 예매 생성
- 예매 조회 및 취소
- 예매 번호 생성

### 3. 결제 (Payment)
- 결제 처리 및 상태 관리
- 환불 처리
- 결제 시뮬레이션

### 4. 유틸리티
- JSON 파싱 (AI 슬롯 데이터)
- QR코드 생성 (예매 완료 후)

## API 엔드포인트

### 기차 조회
- `POST /api/train/search` - 슬롯 데이터로 기차 검색
- `GET /api/train/search` - 파라미터로 기차 검색
- `GET /api/train/{trainNumber}` - 특정 기차 정보 조회

### 예매
- `POST /api/booking` - 예매 생성
- `GET /api/booking/{bookingNumber}` - 예매 조회
- `GET /api/booking` - 전체 예매 조회
- `PUT /api/booking/{bookingNumber}/cancel` - 예매 취소

### 결제
- `POST /api/payment/{bookingId}` - 결제 처리
- `GET /api/payment/{paymentId}` - 결제 조회
- `GET /api/payment/booking/{bookingId}` - 예매별 결제 조회
- `POST /api/payment/{paymentId}/refund` - 환불 처리

## 실행 방법

1. **의존성 설치**
   ```bash
   mvn clean install
   ```

2. **애플리케이션 실행**
   ```bash
   mvn spring-boot:run
   ```

3. **개발 환경 설정**
   - H2 콘솔: http://localhost:8080/h2-console
   - API Base URL: http://localhost:8080/api

## 환경 설정

### 개발 환경 (application-dev.yml)
- H2 인메모리 데이터베이스
- SQL 로깅 활성화
- 디버그 로그 활성화

### 운영 환경 (application-prod.yml)
- MySQL 데이터베이스
- 로그 레벨 최적화
- 보안 설정 강화

## 향후 개발 계획

1. **코레일 API 실제 연동**
2. **카카오톡 API 연동** (예매 정보 전송)
3. **지하철 API 연동**
4. **실제 결제 시스템 연동**
5. **보안 강화** (JWT 토큰, 암호화)
6. **모니터링 및 로깅** (Actuator, Logback)

