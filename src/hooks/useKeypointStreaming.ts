import { useEffect, useRef, useState, useCallback } from 'react';

export interface StreamingOptions {
  enabled: boolean;
  serverUrl: string;
  targetFps?: number; // 전송 FPS (기본 10fps)
  onRecognized?: (label: string, prob: number) => void;
  onError?: (error: Error) => void;
}

export interface StreamingState {
  isConnected: boolean;
  framesSent: number;
  lastError: string | null;
}

/**
 * 키포인트를 서버로 스트리밍하는 Hook
 * WebSocket 우선, 없으면 HTTP POST로 폴백
 */
export function useKeypointStreaming(options: StreamingOptions) {
  const {
    enabled,
    serverUrl,
    targetFps = 10,
    onRecognized,
    onError,
  } = options;

  const [state, setState] = useState<StreamingState>({
    isConnected: false,
    framesSent: 0,
    lastError: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const lastSentTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const sessionIdRef = useRef<string>(
    `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  );

  // WebSocket 연결
  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setState((prev) => ({ ...prev, isConnected: false }));
      return;
    }

    const connectWebSocket = () => {
      try {
        // HTTP URL을 WebSocket URL로 변환
        const wsUrl = serverUrl.replace(/^http/, 'ws');
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log('WebSocket 연결 성공');
          setState((prev) => ({ ...prev, isConnected: true, lastError: null }));
          
          // 초기 메시지 전송 (세션 시작)
          ws.send(
            JSON.stringify({
              type: 'START_SESSION',
              sessionId: sessionIdRef.current,
              timestamp: Date.now(),
            })
          );
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'RESULT' && onRecognized) {
              onRecognized(data.label, data.prob ?? 0);
            }
          } catch (err) {
            console.error('메시지 파싱 실패:', err);
          }
        };

        ws.onerror = (event) => {
          console.error('WebSocket 에러:', event);
          setState((prev) => ({
            ...prev,
            isConnected: false,
            lastError: 'WebSocket 연결 오류',
          }));
          if (onError) {
            onError(new Error('WebSocket 연결 오류'));
          }
        };

        ws.onclose = () => {
          console.log('WebSocket 연결 종료');
          setState((prev) => ({ ...prev, isConnected: false }));
          wsRef.current = null;
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('WebSocket 초기화 실패:', err);
        setState((prev) => ({
          ...prev,
          lastError: err instanceof Error ? err.message : 'WebSocket 초기화 실패',
        }));
        if (onError && err instanceof Error) {
          onError(err);
        }
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, serverUrl, onRecognized, onError]);

  /**
   * 키포인트 프레임 전송
   * FPS 제한 적용
   */
  const sendKeypoints = useCallback(
    (keypoints: number[][]) => {
      if (!enabled || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      // FPS 제한
      const now = Date.now();
      const minInterval = 1000 / targetFps;
      if (now - lastSentTimeRef.current < minInterval) {
        return; // 너무 빠른 전송 스킵
      }

      try {
        const message = {
          type: 'KEYPOINT_FRAME',
          sessionId: sessionIdRef.current,
          timestamp: now,
          frameIndex: frameCountRef.current++,
          keypoints: keypoints,
        };

        wsRef.current.send(JSON.stringify(message));
        lastSentTimeRef.current = now;
        setState((prev) => ({ ...prev, framesSent: prev.framesSent + 1 }));
      } catch (err) {
        console.error('키포인트 전송 실패:', err);
        if (onError && err instanceof Error) {
          onError(err);
        }
      }
    },
    [enabled, targetFps, onError]
  );

  /**
   * HTTP POST로 폴백 전송 (WebSocket 실패 시)
   */
  const sendKeypointsHttp = useCallback(
    async (keypoints: number[][]) => {
      if (!enabled) return;

      const now = Date.now();
      const minInterval = 1000 / targetFps;
      if (now - lastSentTimeRef.current < minInterval) {
        return;
      }

      try {
        const response = await fetch(serverUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sessionId: sessionIdRef.current,
            timestamp: now,
            frameIndex: frameCountRef.current++,
            keypoints: keypoints,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (data.label && onRecognized) {
          onRecognized(data.label, data.prob ?? 0);
        }

        lastSentTimeRef.current = now;
        setState((prev) => ({ ...prev, framesSent: prev.framesSent + 1 }));
      } catch (err) {
        console.error('HTTP 전송 실패:', err);
        if (onError && err instanceof Error) {
          onError(err);
        }
      }
    },
    [enabled, serverUrl, targetFps, onRecognized, onError]
  );

  return {
    sendKeypoints,
    sendKeypointsHttp,
    state,
  };
}

