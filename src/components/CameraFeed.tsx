import { useRef, useEffect } from "react";
import { Holistic, HAND_CONNECTIONS, POSE_CONNECTIONS } from "@mediapipe/holistic";
import type { Results } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

interface CameraFeedProps {
  width?: string;
  height?: string;
  className?: string;
  isRecognizing?: boolean; 
  recognized?: boolean;    
  station?: string | null; 
  // MediaPipe로 추출한 키포인트를 상위 컴포넌트/훅으로 올리고 싶을 때 사용
  onKeypointsCaptured?: (data: {
    poseLandmarks: Results["poseLandmarks"];
    faceLandmarks: Results["faceLandmarks"];
    leftHandLandmarks: Results["leftHandLandmarks"];
    rightHandLandmarks: Results["rightHandLandmarks"];
  }) => void;
}

const CameraFeed = ({
  width = "310px",
  height = "380px",
  className = "",
  onKeypointsCaptured,
}: CameraFeedProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const holisticRef = useRef<Holistic | null>(null);
  const cameraRef = useRef<Camera | null>(null);

  // MediaPipe faceLandmarks(468개)를 OpenPose 70포인트로 축약하기 위한 매핑 테이블
  const FACE_MAPPING_INDICES = useRef<number[]>([
    356, 447, 401, 288, 397, 365, 378, 377, 152, 176, 150, 136, 172, 58, 132, 93, 127, // 0–16 Jawline
    107, 66, 105, 63, 70, // 17–21 Right Eyebrow
    336, 296, 334, 293, 300, // 22–26 Left Eyebrow
    168, 197, 5, 4, // 27–30 Nose bridge
    59, 60, 2, 290, 289, // 31–35 Nose bottom
    33, 160, 158, 133, 153, 163, // 36–41 Right eye
    362, 385, 387, 263, 373, 380, // 42–47 Left eye
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, // 48–59 mouth outer
    78, 81, 13, 311, 308, 402, 14, 178, // 60–67 mouth inner
    1, 2 // 68–69 nose tip points
  ]);
  

  useEffect(() => {
    if (!videoRef.current) return;

    // 1) MediaPipe Holistic 초기화 (얼굴 + 손 + 포즈 모두)
    const holistic = new Holistic({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });


    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults((results: Results) => {
      const canvasEl = canvasRef.current;
      const ctx = canvasEl?.getContext("2d");
      if (canvasEl && ctx) {
        ctx.save();
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        if (results.poseLandmarks) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: "#22c55e",
            lineWidth: 3,
          });
          drawLandmarks(ctx, results.poseLandmarks, {
            color: "#16a34a",
            radius: 2,
          });
        }

        const handStyle = { color: "#f97316", lineWidth: 2 };
        if (results.leftHandLandmarks) {
          drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, handStyle);
          drawLandmarks(ctx, results.leftHandLandmarks, {
            color: "#fb923c",
            radius: 2,
          });
        }
        if (results.rightHandLandmarks) {
          drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, handStyle);
          drawLandmarks(ctx, results.rightHandLandmarks, {
            color: "#fdba74",
            radius: 2,
          });
        }

        ctx.restore();
      }

      // 여기서 얼굴/손/몸 키포인트를 모두 얻을 수 있음
      if (onKeypointsCaptured) {
        const reducedFaceLandmarks =
          results.faceLandmarks && FACE_MAPPING_INDICES.current.length > 0
            ? FACE_MAPPING_INDICES.current
                .map((idx) => results.faceLandmarks && results.faceLandmarks[idx])
                .filter(Boolean)
            : null;

        onKeypointsCaptured({
          poseLandmarks: results.poseLandmarks ?? null,
          faceLandmarks: (reducedFaceLandmarks as Results["faceLandmarks"]) ?? null,
          leftHandLandmarks: results.leftHandLandmarks ?? null,
          rightHandLandmarks: results.rightHandLandmarks ?? null,
        });
      }
    });

    holisticRef.current = holistic;

    //mediapipe holistic의 처리가 카메라 실행보다 일찍 발생하면 오류 발생, 카메라가 실행될 때까지 대기
    const startMediaPipe = async () => {
      try {
          // WASM 모델 로드를 기다림
          await holistic.initialize(); 

          // 2) 모델 초기화 완료 후, Camera 유틸로 웹캠 연결 및 시작
          const camera = new Camera(videoRef.current!, {
              onFrame: async () => {
                  if (!videoRef.current) return;
                  
                  // 비디오 메타데이터가 로드되면 캔버스 크기를 비디오에 맞춤
                  const canvasElement = canvasRef.current;
                  if (canvasElement && videoRef.current) {
                      if (
                          canvasElement.width !== videoRef.current.videoWidth ||
                          canvasElement.height !== videoRef.current.videoHeight
                      ) {
                          canvasElement.width = videoRef.current.videoWidth;
                          canvasElement.height = videoRef.current.videoHeight;
                      }
                  }
                  await holistic.send({ image: videoRef.current });
              },
              width: 640,
              height: 480,
          });

          cameraRef.current = camera;
          camera.start(); // 초기화 완료 후 카메라 스트림 시작

      } catch (error) {
          console.error("MediaPipe 초기화 실패:", error);
          // 초기화 실패 시 사용자에게 알리는 로직 추가 가능
      }
  };
  
  // MediaPipe 실행 시작
  startMediaPipe();

    // 정리: 카메라/모델 리소스 해제
    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
      if (holisticRef.current) {
        holisticRef.current.close();
      }
    };
  }, [onKeypointsCaptured]);

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
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>
  );
};

export default CameraFeed;
