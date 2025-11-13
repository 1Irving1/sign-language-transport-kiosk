import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import CameraFeed from "../components/CameraFeed";
import RecognitionResult from "../components/RecognitionResult";
import RecognitionButtons from "../components/RecognitionButton";

export default function ArrivalPage() {
  const [isRecognizing, setIsRecognizing] = useState(true);
  const [recognized, setRecognized] = useState(false);
  const [station, setStation] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/triptype"); // 도착역 화면으로 이동
  };


// 테스트 
  useEffect(() => {
    if (isRecognizing) {
      const timer = setTimeout(() => {
        setRecognized(true);
        setStation("서울역");
        setIsRecognizing(false);
      }, 4000);

      return () => clearTimeout(timer);
    }
  }, [isRecognizing]);

  return (
    <div className="flex items-center justify-center bg-gray-100">
      <div className="w-[450px] h-[900px] bg-gradient-to-b from-blue-50 to-white shadow-2xl flex flex-col overflow-hidden">
        <Header title="도착역 선택" />

        <main className="flex flex-col flex-1 px-8 overflow-y-auto">
          <div className="mt-6">
            <p className="text-xl text-left font-bold mb-2">
              어느 역으로 가시겠어요?
            </p>
            <p className="text-gray-600 mb-6 text-left">
              도착역 이름을 수어로 표현해주세요.
            </p>

         
            <div className="relative flex items-center justify-center w-full">
              <CameraFeed  
                isRecognizing={isRecognizing}
                recognized={recognized}
                station={station}
              />

          
              {recognized && (
                <div className="absolute bottom-1 left-1/2 -translate-x-1/2 w-[85%]">
                  <RecognitionResult stationName={station!} />
                </div>
              )}
            </div>

          
            {recognized && (
              <div className="flex gap-4 mt-3 justify-center">
                <RecognitionButtons
                  onRetry={() => {
                    setRecognized(false);
                    setIsRecognizing(true);
                    setStation(null);
                  }}
                  onConfirm={handleStart}
                />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
