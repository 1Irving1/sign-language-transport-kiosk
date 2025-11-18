# 수어 인식 기능 설정 가이드

## 개요

프론트엔드에서 MediaPipe Holistic을 사용하여 키포인트를 추출하고,  
실시간으로 추론 서버에 전송하는 구조입니다.

## 아키텍처

```
프론트(React) → 백엔드(Spring Boot) → 추론 서버(Python)
   ↓                    ↓                      ↓
MediaPipe          WebSocket/HTTP          필터링+정규화+모델
키포인트 추출        중계/포워딩              추론 결과 반환
```

## 필요한 패키지 설치

```bash
npm install @mediapipe/holistic @mediapipe/camera_utils
```

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 추가하세요:

```bash
# WebSocket 방식 (권장)
VITE_RECOGNITION_SERVER_URL=ws://localhost:8080/api/sign/stream

# 또는 HTTP POST 방식
# VITE_RECOGNITION_SERVER_URL=http://localhost:8080/api/sign/predict
```

## 파일 구조

```
src/
├── utils/
│   └── mediapipeToOpenpose.ts      # MediaPipe → OpenPose 137점 변환
├── hooks/
│   ├── useHolistic.ts              # MediaPipe Holistic 초기화
│   ├── useKeypointStreaming.ts    # WebSocket/HTTP 스트리밍
│   └── useRecognitionFlow.ts      # 전체 인식 플로우 통합
├── components/
│   └── CameraFeed.tsx              # 카메라 + 인식 상태 표시
└── pages/
    ├── DeparturePage.tsx           # 출발역 인식
    └── ArrivalPage.tsx             # 도착역 인식
```

## 주요 파라미터

### `useRecognitionFlow`

- **serverUrl**: 백엔드 WebSocket/HTTP 주소
- **targetFps**: 전송 FPS (기본 10fps)
- **enableHandFilter**: 손 필터 활성화 (기본 true)
- **onRecognized**: 인식 완료 콜백

### 키포인트 구조

- **137개 키포인트** = 25(body) + 70(face) + 42(hands)
- 각 키포인트: `[x, y, confidence]`

## 사용 예시

```tsx
import { useRecognitionFlow } from '../hooks/useRecognitionFlow';

function DeparturePage() {
  const { videoRef, state, startRecognition, stopRecognition } = useRecognitionFlow({
    serverUrl: 'ws://localhost:8080/api/sign/stream',
    targetFps: 10,
    onRecognized: (label, prob) => {
      console.log(`인식 완료: ${label}, 확률: ${prob}`);
      // 다음 페이지로 이동 등
    },
  });

  useEffect(() => {
    startRecognition();
    return () => stopRecognition();
  }, []);

  return (
    <CameraFeed 
      videoRef={videoRef}
      isRecognizing={state.isRecognizing}
      station={state.recognizedLabel}
    />
  );
}
```

## 서버 응답 포맷

### WebSocket 메시지 (서버 → 클라이언트)

```json
{
  "type": "RESULT",
  "label": "서울역",
  "prob": 0.93
}
```

### HTTP POST 응답

```json
{
  "label": "서울역",
  "prob": 0.93
}
```

## 디버깅

개발 모드(`import.meta.env.DEV`)에서는 화면에 디버그 정보가 표시됩니다:
- 전송된 프레임 수
- 인식 확률

## 트러블슈팅

### 1. MediaPipe 로딩 실패
- CDN 연결 확인: `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/`
- 네트워크 방화벽 확인

### 2. WebSocket 연결 실패
- 백엔드 서버가 실행 중인지 확인
- URL이 `ws://` 또는 `wss://`로 시작하는지 확인
- CORS 설정 확인

### 3. 카메라 접근 거부
- 브라우저 카메라 권한 확인
- HTTPS 환경에서 테스트 (로컬은 HTTP도 가능)

## 성능 최적화

- **FPS 조절**: `targetFps`를 5~15 사이로 조정
- **손 필터**: `enableHandFilter: true`로 불필요한 전송 방지
- **네트워크**: WebSocket 사용 권장 (HTTP보다 오버헤드 적음)

## 다음 단계

1. 백엔드(Spring Boot)에서 WebSocket 엔드포인트 구현
2. 파이썬 추론 서버에서 키포인트 수신/처리 구현
3. 전역 상태 관리(Context/Redux)로 인식 결과 저장
4. 에러 처리 및 폴백 UI 개선

