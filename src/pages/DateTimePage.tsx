import Header from "../components/Header";
import DatePicker from "react-datepicker";
import { useLocation } from "react-router-dom";
import "react-datepicker/dist/react-datepicker.css";
import { ko } from "date-fns/locale/ko";
import "../styles/calendar.css";
import { useState, useEffect } from "react";

export default function DateTimePage() {

  const location = useLocation();
  const tripType = location.state?.tripType ?? "one-way";
  console.log(tripType)

  
  const [step, setStep] = useState<"departure" | "return">("departure");

  // 출발 날짜/시간
  const [departureDate, setDepartureDate] = useState<Date>(new Date());
  const [departureHour, setDepartureHour] = useState<number | null>(null);

  // 복귀 날짜/시간(왕복만)
  const [returnDate, setReturnDate] = useState<Date | null>(null);
  const [returnHour, setReturnHour] = useState<number | null>(null);

  const hours = Array.from({ length: 24 }, (_, i) => i);

  // 디버깅 로그 유지
  useEffect(() => console.log("🚆 가는 날짜:", departureDate), [departureDate]);
  useEffect(() => console.log("⏰ 가는 시간:", departureHour), [departureHour]);
  useEffect(() => console.log("🔄 오는 날짜:", returnDate), [returnDate]);
  useEffect(() => console.log("🔂 오는 시간:", returnHour), [returnHour]);

  
  const handleNext = () => {
    if (tripType === "one-way") {
      console.log("편도 선택 완료 → 바로 기차 조회로 이동");
      // navigate("/trainresult")
      return;
    }

    
    setStep("return");
  };
  
  //기차 조회
  const handleSearchTrain = () => {
    console.log("===== 최종 선택 데이터 =====");
    console.log("여행유형:", tripType);
    console.log("가는날짜:", departureDate);
    console.log("가는시간:", departureHour);
    console.log("오는날짜:", returnDate);
    console.log("오는시간:", returnHour);

    //API 호출
  };



  return (
    <div className="flex justify-center w-screen h-screen bg-white">
      <div className="w-[450px] h-[900px] bg-gradient-to-b from-blue-50 to-white shadow-xl flex flex-col">

        <Header title="날짜/시간 선택" />

        <main className="mt-7 px-6 flex flex-col items-center">

          
          <p className="text-xl font-bold mb-4">
            {step === "departure"
              ? "출발할 날짜와 시간을 선택해주세요."
              : "돌아오는 날짜와 시간을 선택해주세요."}
          </p>

          {/* 공통 달력 · 시간대 UI */}
          <DatePicker
            locale={ko}
            dateFormat="yyyy.MM.dd"
            selected={step === "departure" ? departureDate : returnDate}
            onChange={(d) => {
              if (step === "departure") setDepartureDate(d!);
              else setReturnDate(d!);
            }}
            inline
            calendarClassName="custom-calendar"
            wrapperClassName="custom-calendar-wrapper"
            showTimeSelect={false}
          />

          {/* 시간 선택 */}
          <div className="w-full flex overflow-x-auto gap-3 py-3 no-scrollbar">
            {hours.map((h) => {
              const selected = step === "departure" ? departureHour : returnHour;
              return (
                <button
                  key={h}
                  onClick={() => {
                    if (step === "departure") setDepartureHour(h);
                    else setReturnHour(h);
                  }}
                  className={`flex-shrink-0 px-4 py-2 rounded-xl text-sm border font-semibold
                    ${
                      selected === h
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white text-slate-700 border-slate-300"
                    }
                  `}
                >
                  {String(h).padStart(2, "0")}시
                </button>
              );
            })}
          </div>

          {/* 버튼 */}
          <div className="mt-6">
            {step === "departure" ? (
              <button
                className="bg-blue-600 text-white px-6 py-3 rounded-xl text-lg font-bold"
                onClick={handleNext}
              >
                다음
              </button>
            ) : (
              <button
                className="bg-green-600 text-white px-6 py-3 rounded-xl text-lg font-bold"
                onClick={handleSearchTrain}
              >
                기차 조회하기
              </button>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
