import { useState, useEffect } from "react";
import CameraFeed from "../components/CameraFeed";
import RecognitionResult from "../components/RecognitionResult";
import RecognitionButtons from "../components/RecognitionButton";

export default function DeparturePage() {
  const [isRecognizing, setIsRecognizing] = useState(true);  // 인식 중
  const [recognized, setRecognized] = useState(false);       // 결과 도착 여부
  const [station, setStation] = useState<string | null>(null);

  
  useEffect(() => {
    if (isRecognizing) {
      const timer = setTimeout(() => {
        setRecognized(true);
        setStation("부산");
        setIsRecognizing(false);
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [isRecognizing]);

  return (
    <div className="flex flex-col items-center bg-gradient-to-b from-blue-50 to-white justify-start mt-8">
      <h1 className="text-xl font-bold mb-2">어느 역에서 출발하시겠어요?</h1>
      <p className="text-gray-600 mb-6">출발역 이름을 수어로 표현해주세요.</p>

      {/*<CameraFeed isRecognizing={isRecognizing} recognized={recognized} station={station} /> */}

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
