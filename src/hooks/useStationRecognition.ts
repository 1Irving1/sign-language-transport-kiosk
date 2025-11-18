import { useState, useEffect, useCallback, useRef} from "react";
import { useNavigate } from "react-router-dom";
import { recognizeSignLanguage } from "../api/axios";

// 출발역, 도착역 상태 관리

export type StepType = "departure" | "arrival";

export const useStationRecognition = () => {
  const [step, setStep] = useState<StepType>("departure");
  const [isRecognized, setIsRecognized] = useState(false);
  const navigate = useNavigate();
  const [station, setStation] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsRecognized(true), 6000);
    return () => clearTimeout(timer);
  }, [step]);


  const handleConfirm = async () => {
    try {
      // 가짜 데이터 생성
      const fakeData =
        step === "departure"
          ? JSON.stringify({ gesture: "busan" }) // 출발지 테스트
          : JSON.stringify({ gesture: "seoul" }); // 도착지 테스트

      const result = await recognizeSignLanguage("city", fakeData);

      
      console.log("백엔드 응답:", result);

 
      if (step === "departure") {
        setStep("arrival");
        setIsRecognized(false);
      } else {
        navigate("/triptype");
      }
    } catch (error) {
      console.error("❌ API 호출 실패:", error);
    }
  };





  //!!!!!!!!!!백엔드와 통신!!!!!!!!!!!
  // ⏱️ 마지막으로 요청 보낸 시간을 기억하는 변수
  const lastRequestTimeRef = useRef<number>(0);
  const isProcessingRef = useRef(false);

  const handleRecognition = useCallback(async (data: any) => {
    //인식된 랜드마크들을 JSON으로 표현 후 백으로 전송
    // 1. 이미 결과가 나왔거나, 통신 중이면 스킵
    if (isRecognized || isProcessingRef.current) return;

    // 2. ⭐️ 시간 체크: 1초(1000ms)가 안 지났으면 무시 (핵심)
    const now = Date.now();
    if (now - lastRequestTimeRef.current < 1000) {
        return; 
    }

    // 3. 1초가 지났으므로 전송 시작
    lastRequestTimeRef.current = now; // 시간 갱신
    isProcessingRef.current = true;   // 잠금

    try {
      console.log(`📡 [테스트] 1초 경과: ${step}역 인식 요청 전송...`);
      
      // 백엔드가 List 형태를 받을 수 있으므로, 프레임 하나를 배열에 감싸서 보냄
      const signData = JSON.stringify([data]); 
      
      // API 호출
      const result = await recognizeSignLanguage("city", signData);

      console.log("✅ 백엔드 응답:", result);

      // 응답이 오면 결과 처리
      if (result) {
        setStation(result);
        setIsRecognized(false);  //테스트 이므로 일단 false
      }

    } catch (error) {
      console.error("❌ 연결 테스트 실패:", error);
    } finally {
      isProcessingRef.current = false; // 잠금 해제
    }
  },[isRecognized, step])





  const handleBack = () => {
    if (step === "arrival") setStep("departure");
    else navigate(-1);
  };

  return {
    step,
    isRecognized,
    setIsRecognized,
    handleConfirm,
    handleBack,
    navigate,
    station, 
    handleRecognition, // 👈 이걸 CameraFeed에 줘야 함
  };
};
