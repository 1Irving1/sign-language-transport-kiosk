import { useRef, useEffect } from "react";

interface CameraFeedProps {
  width?: string;
  height?: string;
  className?: string;
  isRecognizing?: boolean; 
  recognized?: boolean;    
  station?: string | null; 
}

const CameraFeed = ({ width = "310px", height = "380px", className = "" }: CameraFeedProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        alert("카메라 접근을 허용해주세요!");
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div
      className={`relative rounded-xl overflow-hidden bg-black shadow-inner ${className}`}
      style={{ width, height }}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />
    </div>
  );
};

export default CameraFeed;
