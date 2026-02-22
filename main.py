import os

# ==========================================
# 🛑 1. macOS HOTFIX 
# ==========================================
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# 🛑 2. GLOBAL IMPORTS (Safe to load here)
# ==========================================
import cv2                  
cv2.setNumThreads(0)        

import mediapipe as mp      # <-- Moved back here!
import tensorflow as tf

# 🔥 Hackathon Hotfix: Patch Keras
original_layer_init = tf.keras.layers.Layer.__init__
def patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_layer_init(self, *args, **kwargs)
tf.keras.layers.Layer.__init__ = patched_layer_init

from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# --- Import Library อื่นๆ ปกติ ---
import json
import numpy as np
import pandas as pd
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Motion Config
MOTION_MODEL_PATH = "Model/motion.keras"
MAX_SEQ_LENGTH = 75
NUM_LANDMARKS = 6
NUM_SENSORS = 6
NUM_BG = 3
MOTION_THRESH_BALANCED = 0.65
MOTION_THRESH_STRICT = 0.85 # Added to prevent NameError in endpoint!

# Vision Config
VISION_MODEL_PATH = "Model/vision/vision.xml"
VISION_THRESH = 0.8

# Fusion Config
FUSION_DECISION_THRESHOLD = 0.5

# ==========================================
# 🧠 CUSTOM LAYER DEFINITION (MOTION)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class WeightedSum(layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.reduce_sum(inputs[0] * inputs[1], axis=1)

    def get_config(self):
        return super(WeightedSum, self).get_config()

# ==========================================
# 🚀 LOAD MODELS (LAZY LOADING C++ HEAVYWEIGHTS)
# ==========================================
motion_model = None
compiled_vision_model = None
vision_output_layer = None
face_detection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global motion_model, compiled_vision_model, vision_output_layer, face_detection
    
    logger.info("⏳ Server starting... Loading C++ Models safely...")
    
    # 1. โหลด MediaPipe (Using the globally imported 'mp')
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    logger.info("✅ 1/3 MediaPipe Loaded.")
    
    # 2. โหลด TensorFlow (Keras)
    motion_model = load_model(MOTION_MODEL_PATH, custom_objects={'WeightedSum': WeightedSum})
    logger.info("✅ 2/3 TensorFlow Motion Model Loaded.")
    
    # 3. โหลด OpenVINO (Lazy Load)
    from openvino.runtime import Core
    ie = Core()
    vision_model_ov = ie.read_model(model=VISION_MODEL_PATH)
    compiled_vision_model = ie.compile_model(model=vision_model_ov, device_name="CPU")
    vision_output_layer = compiled_vision_model.output(0)
    logger.info("✅ 3/3 OpenVINO Vision Model Loaded.")
    
    logger.info("🎉 All models loaded successfully! API is ready.")
    
    yield # This tells FastAPI the startup is done and to start taking requests
    
    logger.info("🛑 Shutting down API...")

# Initialize FastAPI with the lifespan manager
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🏃‍♂️ MOTION PRE-PROCESSING
# ==========================================
def process_signal_data(data_seq):
    dim = data_seq.shape[1] if hasattr(data_seq, 'shape') else 1
    if not isinstance(data_seq, (list, np.ndarray)) or len(data_seq) == 0:
        return np.zeros((MAX_SEQ_LENGTH, dim))
    
    df = pd.DataFrame(data_seq)
    df = df.ewm(span=5, adjust=False).mean()
    vals = df.astype(float).fillna(0.0).values
    
    median = np.median(vals, axis=0)
    q75, q25 = np.percentile(vals, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0 
    normalized = (vals - median) / iqr
    return np.clip(normalized, -4.0, 4.0)

def get_raw_head_pose(lm_matrix):
    """
    lm_matrix mapping จาก Frontend:
    0: จมูก (Nose), 1: ตาซ้าย (L Eye), 2: ตาขวา (R Eye), 
    3: แก้มซ้าย (L Cheek), 4: แก้มขวา (R Cheek), 5: คาง (Chin), 6: หน้าผาก (Forehead)
    """
    try:
        # Yaw: สัดส่วนระยะจมูกถึงแก้มซ้าย เทียบกับระยะจมูกถึงแก้มขวา
        dist_left = np.abs(lm_matrix[0][0] - lm_matrix[3][0])
        dist_right = np.abs(lm_matrix[4][0] - lm_matrix[0][0])
        yaw_ratio = dist_left / (dist_right + 1e-6)
        
        # Pitch: ตำแหน่ง Y ของจมูกเทียบกับระยะหน้าผากถึงคาง
        total_height = np.abs(lm_matrix[5][1] - lm_matrix[6][1]) + 1e-6
        pitch_val = np.abs(lm_matrix[0][1] - lm_matrix[6][1]) / total_height
        
        # Roll: มุมเอียงระหว่างตาซ้ายและตาขวา
        d_x = lm_matrix[2][0] - lm_matrix[1][0]
        d_y = lm_matrix[2][1] - lm_matrix[1][1]
        roll_angle = np.degrees(np.arctan2(d_y, d_x))
        
        return np.array([yaw_ratio, pitch_val, roll_angle])
    except Exception as e:
        return np.array([1.0, 0.5, 0.0])

def prepare_motion_inputs(json_data):
    frames = json_data.get('data', [])
    if len(frames) < 10:
        raise ValueError("Data sequence too short!")

    lm_seq, sn_seq, bg_seq = [], [], []
    prev_lm_coords, prev_pose_vals = None, None
    
    all_vars = [f.get('bg_variance', 0) for f in frames]
    is_gate_open = np.mean(all_vars) >= 3.0 if all_vars else False

    for f in frames:
        raw_lm = f.get('faceMesh')
        if not raw_lm: continue
        
        # Reshape เป็น (7, 3) ให้สอดคล้องกับ 7 Landmarks ที่ส่งมา
        lm_matrix = np.array(raw_lm).reshape(-1, 3) 
        
        # ใช้ตำแหน่ง 0 (Nose Tip) เป็นแกนหลักของการขยับ
        curr_coords = lm_matrix[0].flatten() 
        d_lm = curr_coords - prev_lm_coords if prev_lm_coords is not None else np.zeros(3)
        prev_lm_coords = curr_coords
        
        # คำนวณความเร็วในการหมุนศีรษะ (Delta Pose)
        curr_pose = get_raw_head_pose(lm_matrix)
        d_pose = curr_pose - prev_pose_vals if prev_pose_vals is not None else np.zeros(3)
        prev_pose_vals = curr_pose
        
        # ต่อ Feature การเคลื่อนที่จมูก (3) + การหมุน (3) = 6 มิติ (สอดคล้องกับ NUM_LANDMARKS = 6)
        lm_seq.append(np.concatenate([d_lm, d_pose]))

        s = f.get('sensors', {}) or {}
        a = s.get('accel') or {}; g = s.get('gyro') or {}
        mult = -1 if (f.get('meta') or {}).get('camera_facing') == 'environment' else 1
        sn_seq.append([float(a.get('x',0)), float(a.get('y',0)), float(a.get('z',0))*mult,
                       float(g.get('x',0)), float(g.get('y',0))*mult, float(g.get('z',0))*mult])

        if is_gate_open:
            m = f.get('motion_analysis', {}) or {}
            o = f.get('opticalFlowStats', {}) or {}
            bg_seq.append([float(m.get('relative_magnitude',0)), float(f.get('bg_variance',0)), float(o.get('avgMag',0))])
        else:
            bg_seq.append([0.0, 0.0, 0.0])

    def finalize(arr, target_dim):
        processed = process_signal_data(np.array(arr))
        if len(processed) > MAX_SEQ_LENGTH:
            processed = processed[:MAX_SEQ_LENGTH]
        elif len(processed) < MAX_SEQ_LENGTH:
            processed = np.vstack((processed, np.zeros((MAX_SEQ_LENGTH - len(processed), target_dim))))
        return processed[np.newaxis, ...]

    return [finalize(lm_seq, NUM_LANDMARKS), 
            finalize(sn_seq, NUM_SENSORS), 
            finalize(bg_seq, NUM_BG)]

# ==========================================
# 👁️ VISION PRE-PROCESSING
# ==========================================
def improve_illumination(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def crop_face(image):
    height, width, _ = image.shape
    # Uses the global face_detection initialized in startup
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        x, y = int(bbox.xmin * width), int(bbox.ymin * height)
        w, h = int(bbox.width * width), int(bbox.height * height)
        
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(width, x + w + pad_w), min(height, y + h + pad_h)
        
        face_img = image[y1:y2, x1:x2]
        if face_img.size > 0:
            return face_img
    return image

def process_video_frames_openvino(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        return []

    indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    processed_inputs = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img_processed = crop_face(frame)
            img_processed = improve_illumination(img_processed)
            resized_image = cv2.resize(img_processed, (128, 128))
            
            input_data = np.transpose(resized_image, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            processed_inputs.append(input_data)
    
    cap.release()
    return processed_inputs

# ==========================================
# 🌐 API ENDPOINT
# ==========================================
@app.post("/api/predict/liveness")
async def predict(
    video_file: UploadFile = File(...), 
    json_file: UploadFile = File(...)
):
    if not motion_model or not compiled_vision_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # --- 1. MOTION PREDICTION ---
    motion_score = 0.0
    motion_status = "SPOOF"
    try:
        content = await json_file.read()
        data = json.loads(content)
        
        inputs = prepare_motion_inputs(data)
        preds = motion_model.predict(inputs, verbose=0)
        motion_score = float(preds[0][0])
        
        if motion_score >= MOTION_THRESH_STRICT:
            motion_status = "REAL (Strict)"
        elif motion_score >= MOTION_THRESH_BALANCED:
            motion_status = "REAL (Balanced)"
            
        logger.info(f"🧠 Motion Score: {motion_score:.4f} ({motion_status})")
    except Exception as e:
        logger.error(f"Motion Error: {e}")

    # --- 2. VISION PREDICTION ---
    vision_score = 0.0
    vision_status = "SPOOF"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video_file.read())
            tmp_path = tmp.name
        
        frame_inputs = process_video_frames_openvino(tmp_path)
        os.unlink(tmp_path) 

        if len(frame_inputs) > 0:
            frame_scores = []
            for f_input in frame_inputs:
                res = compiled_vision_model([f_input])[vision_output_layer]
                frame_scores.append(float(res[0][0]))
                
            vision_score = sum(frame_scores) / len(frame_scores)
            
            if vision_score >= VISION_THRESH:
                vision_status = "REAL"
                
            logger.info(f"👁️ Vision Score: {vision_score:.4f} ({vision_status})")
    except Exception as e:
        logger.error(f"Vision Error: {e}")

    # --- 3. FUSION LOGIC ---
    final_score = (vision_score * 0.3) + (motion_score * 0.7)
    is_live = final_score >= FUSION_DECISION_THRESHOLD

    return {
        "final_verdict": "LIVENESS CONFIRMED" if is_live else "LIVENESS DENIED",
        "fusion_score": final_score,
        "details": {
            "motion": {
                "score": motion_score,
                "label": motion_status,
                "thresholds": {"balanced": MOTION_THRESH_BALANCED, "strict": MOTION_THRESH_STRICT}
            },
            "vision": {
                "score": vision_score,
                "label": vision_status,
                "threshold": VISION_THRESH
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)