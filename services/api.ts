import { LivenessData, LivenessApiResponse } from '../types';

// ดึง Base URL จาก Environment Variable (หรือระบุตรงๆ สำหรับทดสอบ)
const API_BASE_URL = 'https://malika-shedable-recollectively.ngrok-free.dev' || process.env.REACT_APP_API_URL ;

export const predictLiveness = async (
  videoBlob: Blob, 
  livenessData: LivenessData
): Promise<LivenessApiResponse> => {
  
  const formData = new FormData();

  // 1. ใส่ไฟล์วิดีโอ (ตั้งชื่อ key ให้ตรงกับ FastAPI: video_file)
  formData.append('video_file', videoBlob, 'capture.mp4');

  // 2. แปลง livenessData เป็น Blob (JSON) และใส่เข้าไป (ชื่อ key: json_file)
  const jsonBlob = new Blob([JSON.stringify(livenessData)], { type: 'application/json' });
  formData.append('json_file', jsonBlob, 'data.json');

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/liveness`, {
      method: 'POST',
      headers: {
        // ✅ สำคัญ: เพื่อข้ามหน้า Browser Warning ของ Ngrok เมื่อเรียกผ่าน Web
        'ngrok-skip-browser-warning': '69420',
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'การส่งข้อมูลไปยังเซิร์ฟเวอร์ล้มเหลว');
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};