# 🔥 Estimating Wildfire Spread Direction from Smoke-Plume Analysis

> **A Wind-Sensor-Free, Physics-Informed Pipeline for UAV Aerial Imagery**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple.svg)](https://docs.ultralytics.com)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)

---

## Abstract

Accurate wildfire spread direction prediction is a life-critical task that traditionally depends on knowing which way the wind is blowing. This project presents a **wind-sensor-free pipeline** that infers wildfire spread direction directly from a single RGB camera or UAV feed.

The system is a four-phase, hierarchical, physics-informed pipeline:
1. **Phase A (Detection)**: Real-time fire and smoke object detection using **YOLOv8s**.
2. **Phase B (Geometry)**: Geometric direction estimation combining a fire-to-smoke displacement vector with PCA plume-axis analysis.
3. **Phase C (Motion)**: Optical-flow confirmation via Farneback dense optical flow (when video is available).
4. **Phase D (Spread Risk)**: Elliptical spread-risk projection following the Rothermel fire-spread model, complete with a confidence-thresholding mechanism to abstain when visual evidence is insufficient.

The result is the first system in the literature that achieves real-time wildfire spread direction prediction from a single RGB camera with zero external wind measurement.

---

## Dataset

The detection backbone is trained on the **FASDD** (Fire And Smoke Detection Dataset), which provides `fire` and `smoke` bounding-box annotations across aerial and ground imagery. The classes are:

| Class | Description |
|---|---|
| **fire** | Visible flame regions |
| **smoke** | Smoke plumes used as passive wind tracers |

---

## Model Architecture & Training

### Architecture
- **Base Model**: YOLOv8s (Small variant) — 11.1 M parameters, 28.4 GFLOPs (Ultralytics 8.4.90)
- **Input Resolution**: 960×960 pixels for training, 640×640 pixels for inference
- **Runtime**: ONNX Runtime (`best.onnx`, dynamic axes) + OpenCV — no PyTorch at inference time
- **Direction Pipeline**: PCA for plume-shape analysis, Farneback for optical flow, circular EMA for temporal smoothing

### Detection Results

Trained on Kaggle. Metrics below are the validation results from the completed 50-epoch `fasdd_train` run (YOLOv8s, imgsz 960, best epoch 50).

| Split | imgsz | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| val | 960 | 0.777 | 0.695 | **0.781** | 0.492 |

**Per-class recall (FASDD test split, from the normalized confusion matrix)**

| Class | Recall | Missed as background |
|---|---|---|
| fire | 0.86 | 0.14 |
| smoke | **0.82** | 0.17 |

*Per-class precision and mAP are pending — capture them from the `model.val()` stdout on Kaggle.*

> **Why the numbers changed:** earlier reports cited mAP@50 = 0.826 with smoke mAP@50 = 0.939. Those weights were trained on a dataset where smoke was largely unlabelled and where video frames leaked across the train/test split, so the smoke figure measured memorisation of a few scenes rather than generalisation — the model scored ~0.006 on smoke for unseen aerial imagery. Retraining on FASDD produces lower but **trustworthy** numbers, and the detector now finds smoke on real UAV photos. See [docs/smoke_defect_report.md](docs/smoke_defect_report.md).

### Inference Throughput (imgsz 640, Tesla T4)

| Metric | Value |
|---|---|
| FPS (mean) | 67.9 |
| FPS (median) | 80.5 |
| Latency (mean) | 14.7 ms |

*Comfortably exceeds the 30 FPS real-time target.*

---

## Visualization

### Direction Estimation Pipeline
The system projects an elliptical spread risk in the expected down-wind region of the frame by fusing the fire→smoke geometry with plume-shape (PCA) cues.

![Spread Direction Estimation](frontend/public/figures/fig_spread.png)

> ⚠️ This figure was generated with the earlier (pre-FASDD) weights and shows a hand-picked, high-confidence subset. Regenerate it from the FASDD model before using it in the report.

### Training and FPS
![Training Metrics](frontend/public/figures/fig_train_and_fps.png)

---

## System Architecture

```
┌─────────────────────────────┐      HTTPS API       ┌──────────────────────────────┐
│   FRONTEND                  │ ◄──────────────────►  │    BACKEND                   │
│                             │                       │                              │
│   Vite + Tailwind CSS v4    │   POST /api/detect/*  │    Flask + ONNX + OpenCV     │
│   Firebase Hosting          │   GET  /api/model-info│    best.onnx (YOLOv8s)       │
│   Static SPA                │                       │    Google Cloud Run          │
└─────────────────────────────┘                       └──────────────────────────────┘
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe for Cloud Run |
| `GET`  | `/api/model-info` | Model metadata, classes, and evaluation metrics |
| `POST` | `/api/detect/image` | Detect fire/smoke + estimate spread direction on an image |
| `POST` | `/api/detect/video` | Same pipeline over a video; returns an annotated MP4 |

**`POST /api/detect/image`** — multipart form: `file` (jpg/png/…), optional `confidence` (0.01–0.9).
Returns JSON:

```json
{
  "annotated_image": "<base64 jpg>",
  "detections": [{"class": "smoke", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}],
  "direction_angle": 87.7,
  "direction_confidence": 0.80,
  "direction_method": "A+B",
  "processing_time": 0.12,
  "image_size": {"width": 640, "height": 480},
  "confidence_threshold": 0.25
}
```

- **`direction_angle`** — estimated spread/wind direction in degrees, math convention (0° = east, 90° = north, CCW). `null` when undetermined.
- **`direction_confidence`** — 0–1. The Phase-D overlay abstains (no arrow/ellipse) below `0.2`.
- **`direction_method`** — which cue produced the estimate: `A` (fire→smoke vector), `B` (plume PCA), `A+B`, `A+B_conflict`, `flow`, or `geom+flow` (video).

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Cloud SDK (for deployment)

### 1. Clone & Setup Export Environment
```bash
git clone <repository-url>
cd dsp-uav

# Create export venv (one-time, for ONNX conversion)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements-export.txt

# Export model to ONNX
python scripts/export_onnx.py
deactivate
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Run locally
python app.py
# API available at http://localhost:5000 (override with the PORT env var)
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# App available at http://localhost:5173
```

---

## License

This project is developed for academic and research purposes (FPT University - DSP391).

---

## Acknowledgments

- **Supervisor**: Mr. Nguyễn Trọng Tài
- **Group 8 Members**: Đỗ Khang, Trần Thoại Các, Nguyễn Minh Triết, Đặng Nguyễn Minh Duy
- [Ultralytics](https://ultralytics.com/) for the YOLOv8 architecture
- [ONNX Runtime](https://onnxruntime.ai/) for optimized inference
- [Kaggle](https://kaggle.com/) for compute resources
