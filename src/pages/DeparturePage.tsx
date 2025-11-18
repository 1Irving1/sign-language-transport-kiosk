import { useState, useEffect, useCallback} from "react";
import type { Results } from "@mediapipe/holistic";
import CameraFeed from "../components/CameraFeed";
import RecognitionResult from "../components/RecognitionResult";
import RecognitionButtons from "../components/RecognitionButton";
import { useStationRecognition } from "../hooks/useStationRecognition";

export default function DeparturePage() {
  const [isRecognizing, setIsRecognizing] = useState(true);  // 인식 중
  const [recognized, setRecognized] = useState(false);       // 결과 도착 여부
  const [station, setStation] = useState<string | null>(null);
  const { 
    handleRecognition, // 👈 이걸 CameraFeed에 줘야 함
  } = useStationRecognition();




  //!!!!!백엔드와 통신!!!!!!!!!
  // MediaPipe Holistic이 추출한 랜드마크를 전달받는 콜백
  const handleKeypointsCaptured = useCallback(async (data: {
    poseLandmarks: Results["poseLandmarks"];
    faceLandmarks: Results["faceLandmarks"];
    leftHandLandmarks: Results["leftHandLandmarks"];
    rightHandLandmarks: Results["rightHandLandmarks"];
  }) => {
    // 현재는 좌표를 콘솔에서 확인만 하고 있음 (향후 서버 전송 등에 활용 가능)
    //console.log("CameraFeed keypoints:", data);

    // 1. 기존 데이터에 direction 필드 추가 (포장)
    const dataWithDirection = {
      ...data,                // 랜드마크 데이터 4개 펼쳐 넣기
      direction: "DEPARTURE"  // 방향 정보 추가
    };
    handleRecognition(dataWithDirection);
  }, [handleRecognition]);





  // useEffect(() => {
  //   if (isRecognizing) {
  //     const timer = setTimeout(() => {
  //       setRecognized(true);
  //       setStation("부산");
  //       setIsRecognizing(false);
  //     }, 4000);
  //     return () => clearTimeout(timer);
  //   }
  // }, [isRecognizing]);

  return (
    <div className="flex flex-col items-center bg-gradient-to-b from-blue-50 to-white justify-start mt-8">
      <h1 className="text-xl font-bold mb-2">어느 역에서 출발하시겠어요?</h1>
      <p className="text-gray-600 mb-6">출발역 이름을 수어로 표현해주세요.</p>

      {/* MediaPipe 기반 CameraFeed에 랜드마크 콜백 연결 */}
      <CameraFeed
        className="mb-6"
        onKeypointsCaptured={handleKeypointsCaptured}
      />

      {recognized && (
        <>
          <RecognitionResult stationName={station!} />
          <RecognitionButtons
            onRetry={() => {
              setRecognized(false);
              setIsRecognizing(true);
              setStation(null);
            }}
            onConfirm={() => alert(`${station}역 출발로 설정되었습니다.`)}
          />
        </>
      )}
    </div>
  );
}
