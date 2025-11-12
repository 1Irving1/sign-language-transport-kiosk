import CameraFeed from "../components/CameraFeed";
import Header
 from "../components/Header";
interface RecognitionLayoutProps {
  title: string;
  subtitle?: string;
  onBack: () => void;
  onHome: () => void;
  isRecognized?: boolean;
  children?: React.ReactNode;
}

const RecognitionLayout = ({
  title,
  subtitle,
  isRecognized,
  children,
}: RecognitionLayoutProps) => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* 상단 헤더 */}
      <Header/>

      {/* 중앙 콘텐츠 */}
      <div className="flex flex-col items-center justify-center flex-grow text-center">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <p className="text-sm text-gray-500 mt-1">{subtitle}</p>

        {/* 카메라 피드 및 인식 결과 */}
        <div className="relative mt-6">
          <CameraFeed />
          {isRecognized && children}
        </div>
      </div>
    </div>
  );
};


export default RecognitionLayout;

