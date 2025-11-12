import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import RecognitionLayout from "../layouts/RecognitionLayout";

// 좌석 종류(일반실, 특실) 인식 페이지

const SeatPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const passengerCount = location.state?.passengerCount || 1;

  const [isRecognized, setIsRecognized] = useState(false);
  const [seatType, setSeatType] = useState<"일반실" | "특실" | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setSeatType("일반실"); 
      setIsRecognized(true);
    }, 5000);
    return () => clearTimeout(timer);
  }, []);

 
  const handleConfirm = () => {
    navigate("/seatlist", {
      state: {
        seatType,
        passengerCount,
      },
    });
  };

  const handleRetry = () => {
    setIsRecognized(false);
    setSeatType(null);
  };

  return (
    <RecognitionLayout
      title="어떤 좌석을 이용하시겠어요?"
      subtitle="일반실 또는 특실을 수어로 표현해주세요."
      stationName={
        isRecognized && seatType ? `${seatType} 인식 완료` : ""
      }
      isRecognized={isRecognized}
      onRetry={handleRetry}
      onConfirm={handleConfirm}
      onBack={() => navigate(-1)}
      onHome={() => navigate("/")}
    />
  );
};

export default SeatPage;
