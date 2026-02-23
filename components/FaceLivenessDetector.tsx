
import React, { useState } from 'react';
import FaceScanner from './FaceScanner';
import ResultsDisplay from './ResultsDisplay';
import { predictLiveness } from '../services/api';
import { LivenessData, LivenessApiResponse } from '../types';
import { LogoIcon, ShieldCheckIcon, ShieldExclamationIcon } from './icons';
import Spinner from './Spinner';


type AppState = 'idle' | 'scanning' | 'confirm' | 'loading' | 'results' | 'error';

const FaceLivenessDetector: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('idle');
  const [livenessData, setLivenessData] = useState<LivenessData | null>(null);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [results, setResults] = useState<LivenessApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleScanComplete = (data: LivenessData, blob: Blob) => {
    setLivenessData(data);
    setVideoBlob(blob);
    setAppState('confirm');
  };

  const handleConfirm = async () => {
    if (!livenessData || !videoBlob) {
      setError('Captured data is missing.');
      setAppState('error');
      return;
    }

    setAppState('loading');
    setError(null);
    setResults(null);

    try {
      const apiResults = await predictLiveness(videoBlob, livenessData);
      setResults(apiResults);
      setAppState('results');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(errorMessage);
      setAppState('error');
    }
  };

  const handleReset = () => {
    setAppState('idle');
    setLivenessData(null);
    setVideoBlob(null);
    setResults(null);
    setError(null);
  };
  
  const handleStartScan = async () => {
    // For iOS, permission must be requested on a user gesture.
    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      try {
        await (DeviceMotionEvent as any).requestPermission();
      } catch (error) {
        // User may have denied, or an error occurred.
        // The app will proceed, and the scanner will use dummy data if sensors are unavailable.
        console.warn("Device motion permission request failed or was denied.", error);
      }
    }
    setAppState('scanning');
  };

  const renderContent = () => {
    switch (appState) {
      case 'scanning':
        return <FaceScanner onScanComplete={handleScanComplete} onCancel={() => setAppState('idle')} />;
      case 'confirm':
        return (
          <div className="text-center flex flex-col items-center gap-6">
            <ShieldCheckIcon className="w-24 h-24 text-green-400" />
            <h2 className="text-2xl font-bold">Scan Complete</h2>
            <p className="text-gray-400 max-w-sm">
              Face and motion data captured successfully. Press confirm to upload and analyze the data for liveness verification.
            </p>
            <button
              onClick={handleConfirm}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
            >
              Confirm & Analyze
            </button>
             <button
              onClick={handleReset}
              className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition"
            >
              Retry Scan
            </button>
          </div>
        );
      case 'loading':
        return (
          <div className="text-center flex flex-col items-center gap-4">
            <Spinner />
            <h2 className="text-2xl font-bold animate-pulse">Analyzing Liveness...</h2>
            <p className="text-gray-400 max-w-sm">
              Please wait. Our AI is processing both motion and video data to ensure security.
            </p>
          </div>
        );
      case 'results':
        return results && <ResultsDisplay results={results} onReset={handleReset} />;
      case 'error':
        return (
            <div className="text-center flex flex-col items-center gap-6">
            <ShieldExclamationIcon className="w-24 h-24 text-red-400" />
            <h2 className="text-2xl font-bold">An Error Occurred</h2>
            <p className="text-red-400 bg-red-900/50 p-4 rounded-lg max-w-md">
                {error}
            </p>
            <button
              onClick={handleReset}
              className="w-full max-w-sm bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
            >
              Try Again
            </button>
          </div>
        );
      case 'idle':
      default:
        return (
          <div className="text-center flex flex-col items-center gap-6">
            <LogoIcon className="w-24 h-24 text-indigo-400" />
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">GSYNC Liveness Detection</h1>
            <p className="text-gray-400 max-w-md">
              [DEMO] ระบบ Liveness Detection ที่ใช้ AI วิเคราะห์ความสอดคล้องระหว่างใบหน้าและเซนเซอร์มือถือ
            </p>
            <button
              onClick={handleStartScan}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-4 px-6 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 text-lg"
            >
              Start Liveness Scan
            </button>
          </div>
        );
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-6 sm:p-8 border border-gray-700">
      {renderContent()}
    </div>
  );
};

export default FaceLivenessDetector;
