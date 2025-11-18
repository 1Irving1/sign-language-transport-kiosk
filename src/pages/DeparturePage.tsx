import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import CameraFeed from "../components/CameraFeed";
import RecognitionResult from "../components/RecognitionResult";
import Header from "../components/Header";
import RecognitionButtons from "../components/RecognitionButton";
import { useRecognitionFlow } from "../hooks/useRecognitionFlow";

export default function DeparturePage() {
  const navigate = useNavigate();
  
  // 실제 인식 플로우 사용
  const { videoRef, state, startRecognition, stopRecognition, resetResult } = useRecognitionFlow({
    serverUrl: import.meta.env.VITE_RECOGNITION_SERVER_URL || 'ws://localhost:8080/api/sign/stream',
    targetFps: 10,
    enableHandFilter: true,
    onRecognized: (label, prob) => {
      console.log(`출발역 인식 완료: ${label} (확률: ${prob})`);
      // 인식 완료 시 자동으로 다음 페이지로 이동 (옵션)
      // setTimeout(() => navigate('/arrival'), 2000);
    },
  });

  // 페이지 진입 시 인식 자동 시작
  useEffect(() => {
    startRecognition();
    
    return () => {
      stopRecognition();
    };
  }, []);

  const handleRetry = () => {
    resetResult();
    startRecognition();
  };

  const handleConfirm = () => {
    if (state.recognizedLabel) {
      // 전역 상태에 저장 (Context 등 사용 시)
      // setGlobalDepartureStation(state.recognizedLabel);
      
      // 다음 페이지로 이동
      navigate('/arrival');
    }
  };

  return (
    <div className="flex flex-col items-center bg-gradient-to-b from-blue-50 to-white justify-start mt-8">
      <h1 className="text-xl font-bold mb-2">어느 역에서 출발하시겠어요?</h1>
      <p className="text-gray-600 mb-6">출발역 이름을 수어로 표현해주세요.</p>

      {/* 에러 표시 */}
      {state.error && (
        <div className="bg-red-100 text-red-700 px-4 py-2 rounded-lg mb-4">
          ⚠️ {state.error}
        </div>
      )}

      {/* 연결 상태 표시 */}
      {!state.isReady && state.isRecognizing && (
        <div className="bg-yellow-100 text-yellow-700 px-4 py-2 rounded-lg mb-4">
          인식 서버 연결 중...
        </div>
      )}

      {/* 카메라 피드 영역 */}
      <div className="relative flex items-center justify-center w-full mb-4">
        <CameraFeed 
          videoRef={videoRef}
          isRecognizing={state.isRecognizing} 
          recognized={state.recognizedLabel !== null} 
          station={state.recognizedLabel}
        />

        {/* 인식 결과를 카메라 하단에 표시 */}
        {state.recognizedLabel && (
          <div className="absolute bottom-1 left-1/2 -translate-x-1/2 w-[85%]">
            <RecognitionResult stationName={state.recognizedLabel} />
          </div>
        )}
      </div>

      {/* 디버그 정보 (개발 시) */}
      {import.meta.env.DEV && (
        <div className="text-xs text-gray-500 mb-2">
          전송 프레임: {state.framesSent} | 확률: {state.recognizedProb?.toFixed(2) || 'N/A'}
        </div>
      )}

      {/* 재시도 및 확인 버튼 */}
      {state.recognizedLabel && (
        <div className="flex gap-4 mt-3 justify-center">
          <RecognitionButtons
            onRetry={handleRetry}
            onConfirm={handleConfirm}
          />
        </div>
      )}
    </div>
  );
}
