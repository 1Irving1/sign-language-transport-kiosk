import Header from "../components/Header";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import 
{ko} from "date-fns/locale/ko"; 
import "../styles/calendar.css";
import { useState } from "react";

export default function DateTimePage() {
  const [currentDate, setCurrentDate] = useState<Date>(new Date());
  const [selectedHour, setSelectedHour] = useState<number | null>(null);

  
  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="flex justify-center w-screen h-screen bg-white">
      <div className="w-[450px] h-[900px] bg-gradient-to-b from-blue-50 to-white shadow-xl flex flex-col">
        
        <Header title="날짜/시간 선택" />

        <main className="mt-7 px-6 flex flex-col items-center">
          <p className="text-xl font-bold mb-4">날짜와 시간을 선택해주세요</p>

          {/* 달력 */}
          <DatePicker
            locale={ko}
            dateFormat="yyyy.MM.dd"
            selected={currentDate}
            onChange={(date) => setCurrentDate(date!)}
            inline
            calendarClassName="custom-calendar"
            wrapperClassName="custom-calendar-wrapper"
            showTimeSelect={false}  
          />

          {/* 시간 선택 */}
          <p className="text-xl font-bold mt-6 mb-3"></p>

          <div className="w-full flex overflow-x-auto gap-3 py-3 no-scrollbar">
            {hours.map((h) => (
              <button
                key={h}
                onClick={() => setSelectedHour(h)}
                className={`
                  flex-shrink-0 px-4 py-2 rounded-xl text-sm border font-semibold
                  ${selectedHour === h
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-slate-700 border-slate-300"}
                `}
              >
                {String(h).padStart(2, "0")}시
              </button>
            ))}
          </div>

        </main>
      </div>
    </div>
  );
}
