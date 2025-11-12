import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import RecognitionLayout from "../layouts/RecognitionLayout";

// 탑승 인원 페이지
const PassengerPage = () => {
  const navigate = useNavigate();
  const [isRecognized, setIsRecognized] = useState(false);
  const [recognizedCount, setRecognizedCount] = useState<number | null>(null);

 
  useEffect(() => {
    const timer = setTimeout(() => {
      setRecognizedCount(2); // 2명 인식
      setIsRecognized(true);
    }, 5000);
    return () => clearTimeout(timer);
  }, []);

  const handleConfirm = () => {
    navigate("/seat", { state: { passengerCount: recognizedCount } });
  };

  const handleRetry = () => {
    setIsRecognized(false);
    setRecognizedCount(null);
  };

  return (
    <RecognitionLayout
      title="승차 인원을 알려주세요."
      subtitle="탑승하실 인원을 수어로 표현해주세요."
      stationName={
        isRecognized && recognizedCount !== null
          ? `승차 인원: ${recognizedCount}명 인식 완료`
          : ""
      }
      isRecognized={isRecognized}
      onRetry={handleRetry}
      onConfirm={handleConfirm}
      onBack={() => navigate(-1)}
      onHome={() => navigate("/")}
    />
  );
};

export default PassengerPage;