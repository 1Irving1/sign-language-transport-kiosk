import RecognitionLayout from "../layouts/RecognitionLayout";
import { useStationRecognition } from "../hooks/useStationRecognition";

const StationPage = () => {
  const { step, isRecognized, setIsRecognized, handleConfirm, handleBack, navigate } =
    useStationRecognition();

  const title =
    step === "departure"
      ? "어느 역에서 출발하시겠어요?"
      : "어느 역으로 가시겠어요?";
  const subtitle =
    step === "departure"
      ? "출발역 이름을 수어로 표현해주세요."
      : "도착역 이름을 수어로 표현해주세요.";

  return (
    <RecognitionLayout
      title={title}
      subtitle={subtitle}
      stationName={step === "departure" ? "부산" : "서울"}
      isRecognized={isRecognized}
      onRetry={() => setIsRecognized(false)}
      onConfirm={handleConfirm}
      onBack={handleBack}
      onHome={() => navigate("/")}
    />
  );
};

export default StationPage;
