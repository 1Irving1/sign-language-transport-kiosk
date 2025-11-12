interface RecognitionButtonsProps {
  onRetry: () => void;
  onConfirm: () => void;
}

export default function RecognitionButtons({ onRetry, onConfirm }: RecognitionButtonsProps) {
  return (
    <div className="flex gap-4">
      <button
        onClick={onRetry}
        className="px-5 py-2.5 bg-white border border-sky-300 text-sky-600 font-semibold rounded-xl 
                   hover:bg-sky-50 active:scale-95 transition-all duration-150"
      >
        다시 인식하기
      </button>
      <button
        onClick={onConfirm}
        className="px-6 py-2.5 bg-sky-500 text-white font-semibold rounded-xl 
                   hover:bg-sky-600 active:scale-95 transition-all duration-150"
      >
        맞아요
      </button>
    </div>
  );
}
