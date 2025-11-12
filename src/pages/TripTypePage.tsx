import RecognitionLayout from "../layouts/RecognitionLayout";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const TripTypePage = () => {
  const navigate = useNavigate();
  const [isRecognized, setIsRecognized] = useState(false);
  const [tripType, setTripType] = useState<"oneway" | "round" | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setTripType("oneway"); // 나중에 실제 AI 결과로 교체
      setIsRecognized(true);
    }, 6000);
    return () => clearTimeout(timer);
  }, []);

  const handleConfirm = () => {
    navigate("/datetime", { state: { tripType } }); 
  };

  const handleRetry = () => {
    setIsRecognized(false);
    setTripType(null);
  };

  return (
    <RecognitionLayout
      title="편도로 예매하시겠어요, 왕복으로 하시겠어요?"
      subtitle="수어로 편도 또는 왕복을 표현해주세요."
      stationName={tripType === "oneway" ? "편도" : "왕복"}
      isRecognized={isRecognized}
      onRetry={handleRetry}
      onConfirm={handleConfirm}
      onBack={() => navigate(-1)}
      onHome={() => navigate("/")}
    />
  );
};

export default TripTypePage;
