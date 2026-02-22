import React from 'react';
import { LivenessApiResponse } from '../types';
import { CheckCircleIcon, XCircleIcon, ShieldCheckIcon, ShieldExclamationIcon } from './icons';

interface ResultsDisplayProps {
  results: LivenessApiResponse;
  onReset: () => void;
}

const ResultCard: React.FC<{ title: string; label: string; score: number; isSuccess: boolean }> = ({ title, label, score, isSuccess }) => (
  <div className={`p-4 rounded-lg border ${isSuccess ? 'bg-green-900/50 border-green-700' : 'bg-red-900/50 border-red-700'}`}>
    <h3 className="text-lg font-semibold text-gray-300">{title}</h3>
    <div className="flex items-center justify-between mt-2">
      <div className="flex items-center gap-2">
        {isSuccess ? <CheckCircleIcon className="w-6 h-6 text-green-400" /> : <XCircleIcon className="w-6 h-6 text-red-400" />}
        <span className={`text-xl font-bold ${isSuccess ? 'text-green-300' : 'text-red-300'}`}>{label}</span>
      </div>
      <span className="text-sm font-mono px-2 py-1 bg-gray-700 rounded">{score.toFixed(4)}</span>
    </div>
  </div>
);

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, onReset }) => {
  const isFinalSuccess = results.final_verdict === 'LIVENESS CONFIRMED';
  
  // ✅ แก้ไข: เปลี่ยนจาก 'LIVE' เป็น 'REAL' ให้ตรงกับ main.py
  const motionSuccess = results.details?.motion?.label === 'REAL';
  const visionSuccess = results.details?.vision?.label === 'REAL'; 

  return (
    <div className="w-full flex flex-col items-center gap-6 text-center">
      {isFinalSuccess ? (
        <ShieldCheckIcon className="w-24 h-24 text-green-400" />
      ) : (
        <ShieldExclamationIcon className="w-24 h-24 text-red-400" />
      )}
      
      <div className="space-y-1">
        <h2 className={`text-3xl font-extrabold ${isFinalSuccess ? 'text-green-300' : 'text-red-300'}`}>
          {results.final_verdict}
        </h2>
        {/* แสดงคะแนน Fusion Score รวม */}
        {results.fusion_score !== undefined && (
          <p className="text-sm font-mono text-gray-400">
            Fusion Score: {results.fusion_score.toFixed(4)}
          </p>
        )}
      </div>

      <div className="w-full space-y-4 text-left">
        <ResultCard 
          title="Motion Analysis"
          label={results.details?.motion?.label || 'UNKNOWN'}
          score={results.details?.motion?.score || 0}
          isSuccess={motionSuccess}
        />
        <ResultCard 
          title="Vision Analysis"
          label={results.details?.vision?.label || 'UNKNOWN'}
          score={results.details?.vision?.score || 0}
          isSuccess={visionSuccess}
        />
      </div>

      <p className="text-gray-400">
        {isFinalSuccess 
          ? "Liveness verified. Both motion and vision analysis passed." 
          : "Liveness could not be verified. One or more checks failed."}
      </p>

      <button
        onClick={onReset}
        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 ease-in-out"
      >
        Scan Again
      </button>
    </div>
  );
};

export default ResultsDisplay;