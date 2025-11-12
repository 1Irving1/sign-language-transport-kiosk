import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom"; 
import RecognitionLayout from "../layouts/RecognitionLayout";

type Step =
  | "departureDate"
  | "departureTime"
  | "returnDate"
  | "returnTime";

export default function DateTimePage() {
  const navigate = useNavigate();
  const location = useLocation(); 
  const tripType = location.state?.tripType || "oneway"; 

  const [step, setStep] = useState<Step>("departureDate");
  const [isRecognized, setIsRecognized] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => {
      if (step === "departureDate") setRecognizedText("10월 3일");
      if (step === "departureTime") setRecognizedText("오후 1시 - 2시");
      if (step === "returnDate") setRecognizedText("10월 7일");
      if (step === "returnTime") setRecognizedText("오전 11시");
      setIsRecognized(true);
    }, 5000);
    return () => clearTimeout(timer);
  }, [step]);

  const handleConfirm = () => {
    if (step === "departureDate") {
      setStep("departureTime");
      setIsRecognized(false);
    } else if (step === "departureTime" && tripType === "round") {
      setStep("returnDate");
      setIsRecognized(false);
    } else if (step === "returnDate") {
      setStep("returnTime");
      setIsRecognized(false);
    } else {
      navigate("/seat"); 
    }
  };

  const titleMap: Record<Step, string> = {
    departureDate: "언제 출발하시겠어요?",
    departureTime: "출발 시간은 언제인가요?",
    returnDate: "돌아오실 날짜를 알려주세요.",
    returnTime: "돌아오실 시간을 알려주세요.",
  };

  const subtitleMap: Record<Step, string> = {
    departureDate: "출발 날짜와 시간을 수어로 표현해주세요.",
    departureTime: "출발 시간을 수어로 표현해주세요.",
    returnDate: "복귀 날짜와 시간을 수어로 표현해주세요.",
    returnTime: "복귀 시간을 수어로 표현해주세요.",
  };

  return (
    <RecognitionLayout
      title={titleMap[step]}
      subtitle={subtitleMap[step]}
      stationName={`${recognizedText} 인식 완료`}
      isRecognized={isRecognized}
      onRetry={() => setIsRecognized(false)}
      onConfirm={handleConfirm}
      onBack={() => navigate(-1)}
      onHome={() => navigate("/")}
    />
  );
}
