import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { SensorData } from '../types'; 

interface FaceScannerProps {
  onScanComplete: (data: any, blob: Blob) => void;
  onCancel: () => void;
}

const MAX_FRAMES = 120;
const SELECTED_LANDMARKS = [1, 33, 263, 234, 454, 152, 10];

const FaceScanner: React.FC<FaceScannerProps> = ({ onScanComplete, onCancel }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const collectedFramesRef = useRef<any[]>([]); 
  const isRecordingRef = useRef(false);
  const animationFrameIdRef = useRef<number | null>(null);
  const detectorRef = useRef<faceLandmarksDetection.FaceLandmarksDetector | null>(null);
  
  const [status, setStatus] = useState('Initializing AI...');
  const [progress, setProgress] = useState(0);

  const sensorRef = useRef<SensorData>({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
  });

  // --- 1. SENSOR SETUP ---
  useEffect(() => {
    const handleMotion = (event: DeviceMotionEvent) => {
      if (event.accelerationIncludingGravity) {
        sensorRef.current.accel = {
          x: event.accelerationIncludingGravity.x || 0,
          y: event.accelerationIncludingGravity.y || 0,
          z: event.accelerationIncludingGravity.z || 0,
        };
      }
      if (event.rotationRate) {
        sensorRef.current.gyro = {
          x: event.rotationRate.alpha || 0,
          y: event.rotationRate.beta || 0,
          z: event.rotationRate.gamma || 0,
        };
      }
    };

    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      (DeviceMotionEvent as any).requestPermission().then((state: string) => {
        if (state === 'granted') window.addEventListener('devicemotion', handleMotion);
      });
    } else {
      window.addEventListener('devicemotion', handleMotion);
    }

    return () => window.removeEventListener('devicemotion', handleMotion);
  }, []);

  // --- 2. FAST INITIALIZATION LOGIC ---
  useEffect(() => {
    let active = true;

    const setupApp = async () => {
      try {
        const [_, stream] = await Promise.all([
          tf.setBackend('webgl').then(() => tf.ready()),
          navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, 
            audio: false 
          })
        ]);

        if (!active) return;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await new Promise<void>((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadeddata = () => {
                // ✅ สำคัญที่ 1: ตั้งค่า Canvas ให้ความละเอียดเท่ากล้องเป๊ะๆ
                if (canvasRef.current && videoRef.current) {
                  canvasRef.current.width = videoRef.current.videoWidth;
                  canvasRef.current.height = videoRef.current.videoHeight;
                }
                resolve();
              };
            } else {
              resolve();
            }
          });
        }

        const detector = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          {
            runtime: 'mediapipe', 
            solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh`,
            refineLandmarks: true,
            maxFaces: 1,
          }
        );

        if (!active) return;
        detectorRef.current = detector;

        if (videoRef.current && videoRef.current.videoWidth > 0) {
            await detector.estimateFaces(videoRef.current);
        }

        setStatus('Scanning...');
        startRecording(stream);
        scanFrame();

      } catch (err) {
        console.error("Initialization failed:", err);
        setStatus('Error: Camera/AI not ready');
      }
    };

    setupApp();
    return () => { active = false; };
  }, []);

  const startRecording = (stream: MediaStream) => {
    isRecordingRef.current = true;
    recordedChunksRef.current = [];
    collectedFramesRef.current = []; 

    const options = { mimeType: 'video/mp4' };
    try {
        mediaRecorderRef.current = new MediaRecorder(stream, MediaRecorder.isTypeSupported('video/mp4') ? options : undefined);
    } catch (e) {
        mediaRecorderRef.current = new MediaRecorder(stream);
    }

    mediaRecorderRef.current.ondataavailable = (e) => e.data.size > 0 && recordedChunksRef.current.push(e.data);
    mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/mp4' });
        const finalData = {
            type: "LIVENESS_SCAN",
            scenario: "Production",
            data: collectedFramesRef.current,
            meta: { 
                userAgent: navigator.userAgent,
                camera_facing: 'user' // ส่งค่านี้เพื่อให้ Backend ปรับคูณ -1 สำหรับ Gyro/Accel ได้ถูกต้อง
            }
        };
        onScanComplete(finalData, blob);
    };
    mediaRecorderRef.current.start();
  };

  const scanFrame = async () => {
    if (!videoRef.current || !canvasRef.current || !isRecordingRef.current || !detectorRef.current || videoRef.current.videoWidth === 0) {
        animationFrameIdRef.current = requestAnimationFrame(scanFrame);
        return;
    }

    try {
        const faces = await detectorRef.current.estimateFaces(videoRef.current, { flipHorizontal: false });
        const ctx = canvasRef.current.getContext('2d');
        
        if (ctx && faces.length > 0) {
            const face = faces[0];
            const width = videoRef.current.videoWidth;
            const height = videoRef.current.videoHeight;
            
            // ✅ สำคัญที่ 2: ใช้ตัวแปร width, height แทนการระบุเลข 640, 480
            ctx.clearRect(0, 0, width, height);
            ctx.save();

            const flatFaceMesh: number[] = [];
            SELECTED_LANDMARKS.forEach(index => {
                const p = face.keypoints[index];
                flatFaceMesh.push(p.x / width, p.y / height, (p as any).z ? (p as any).z / width : 0);
                
                ctx.fillStyle = '#6366f1';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
                ctx.fill();
            });

            collectedFramesRef.current.push({
                timestamp: Date.now(),
                faceMesh: flatFaceMesh,
                sensors: { ...sensorRef.current }
            });

            const progressVal = (collectedFramesRef.current.length / MAX_FRAMES) * 100;
            setProgress(progressVal);

            if (collectedFramesRef.current.length >= MAX_FRAMES) {
                isRecordingRef.current = false;
                mediaRecorderRef.current?.stop();
                return; 
            }
            ctx.restore();
        } else if (ctx) {
            // เคลียร์หน้าจอถ้าหาหน้าไม่เจอ
            const width = videoRef.current.videoWidth;
            const height = videoRef.current.videoHeight;
            ctx.clearRect(0, 0, width, height);
        }
    } catch (e) {
        console.warn("Frame skipped due to error: ", e);
    }

    animationFrameIdRef.current = requestAnimationFrame(scanFrame);
  };

  useEffect(() => {
    return () => {
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="relative w-full max-w-sm aspect-[3/4] rounded-lg overflow-hidden shadow-2xl border-2 border-indigo-500 bg-black">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" 
        />
        <canvas 
          ref={canvasRef} 
          className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" 
        />
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-6 text-center">
          <p className="font-bold text-xl text-white mb-3 tracking-wide">{status}</p>
          <div className="w-full bg-gray-800/50 rounded-full h-2.5 backdrop-blur-md overflow-hidden">
            <div 
                className="bg-indigo-500 h-full transition-all duration-150 ease-out" 
                style={{ width: `${progress}%` }} 
            />
          </div>
          <p className="text-xs text-indigo-300 mt-2 font-mono">{Math.round(progress)}% COMPLETE</p>
        </div>
      </div>
      <button onClick={onCancel} className="px-6 py-2 text-gray-400 hover:text-white transition-colors">Cancel</button>
    </div>
  );
};

export default FaceScanner;