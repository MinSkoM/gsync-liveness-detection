# GSYNC — Face Liveness Detection

A full-stack face liveness detection system that fuses **motion signals** (FaceMesh + IMU sensors + optical flow) with **visual analysis** (CNN) to determine whether a face in front of the camera belongs to a live person or a spoof attempt.

Built as part of the **GSync Liveness Detection** project at KBTG.

---

## How it works

```
User scans face (browser)
        │
        ├─ FaceMesh landmarks (MediaPipe)
        ├─ IMU sensor data (accelerometer + gyroscope)
        ├─ Background optical flow (OpenCV.js)
        └─ Video frames
        │
        ▼
FastAPI backend
        ├─ Motion Model (TensorFlow/Keras)  ─── 70% weight ─┐
        └─ Vision Model (OpenVINO CNN)       ─── 30% weight ─┤
                                                              ▼
                                                    Fusion Score → LIVENESS CONFIRMED / DENIED
```

### Motion Model
- Input: 75-frame sequence of per-frame features
  - **FaceMesh delta** — 3D nose-tip movement (Δx, Δy, Δz)
  - **Head pose delta** — yaw ratio, pitch, roll computed from 7 key landmarks
  - **IMU** — accelerometer (x, y, z) + gyroscope (α, β, γ), EWM-smoothed and IQR-normalized
  - **Optical flow** — background relative motion magnitude, variance, average magnitude (gated by scene motion threshold)
- Custom `WeightedSum` attention layer
- Threshold: `0.65` (balanced) / `0.50` (strict)

### Vision Model
- Samples 5 frames evenly from the recorded video
- CLAHE illumination enhancement → resize to 128×128
- OpenVINO-compiled CNN inference on CPU
- Threshold: `0.8`

### Fusion
```
final_score = (vision_score × 0.3) + (motion_score × 0.7)
```
Decision threshold: `0.5`

---

## Architecture

```
browser (React + TypeScript + Vite)
  └── FaceScanner         — records video + FaceMesh + IMU + optical flow
  └── FaceLivenessDetector — scan → confirm → analyze → results flow
  └── ResultsDisplay      — renders verdict and per-model scores

backend (FastAPI + Python)
  └── POST /api/predict/liveness
        ├── video_file   (multipart, .mp4)
        └── json_file    (multipart, LivenessData JSON)
  └── Models/
        ├── motion.keras          (TensorFlow/Keras)
        └── vision/vision.xml     (OpenVINO IR format)
```

---

## Setup

**Prerequisites:** Node.js, Python 3.10+

### Frontend

```bash
npm install
npm run dev
```

Set the backend URL in `.env.local`:

```
VITE_API_URL=http://<your-server-ip>:8000
```

### Backend

```bash
pip install -r requirements_backup.txt
# or with Pipenv
pipenv install

python main.py
```

The server starts on `0.0.0.0:8000`. Models are loaded lazily at startup via FastAPI `lifespan`.

Place model files at:
```
Model/
  motion.keras
  vision/
    vision.xml
    vision.bin
```

---

## API

### `POST /api/predict/liveness`

**Multipart form fields:**

| Field | Type | Description |
|---|---|---|
| `video_file` | file (.mp4) | Recorded video from the scanner |
| `json_file` | file (.json) | `LivenessData` — frame-by-frame FaceMesh, IMU, optical flow |

**Response:**

```json
{
  "final_verdict": "LIVENESS CONFIRMED",
  "fusion_score": 0.81,
  "details": {
    "motion": { "score": 0.92, "label": "REAL", "thresholds": { "balanced": 0.65 } },
    "vision": { "score": 0.55, "label": "SPOOF", "threshold": 0.8 }
  }
}
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| Face tracking | MediaPipe FaceMesh (browser, via CDN) |
| Optical flow | OpenCV.js (WASM) |
| IMU | Device Motion API |
| Backend | FastAPI, Uvicorn |
| Motion model | TensorFlow / Keras |
| Vision model | OpenVINO Runtime |
| Face detection | MediaPipe Face Detection (Python) |

---

## Notes

- On iOS, the app requests `DeviceMotionEvent` permission before scanning starts.
- Rear-camera recordings invert the accelerometer X axis so left/right direction stays consistent with front-camera coordinate frame.
- The backend applies a **background motion gate**: if average `bg_variance < 3.0` across all frames, optical flow features are zeroed out to avoid noise from static scenes.
- macOS-specific environment flags (`OBJC_DISABLE_INITIALIZE_FORK_SAFETY`, `OMP_NUM_THREADS`, etc.) are set in `main.py` to prevent forking issues with TensorFlow and OpenCV.
