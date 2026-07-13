# Estimating Wildfire Spread Direction from Smoke-Plume Analysis: A Wind-Sensor-Free, Physics-Informed Pipeline

## Background & Project Research Findings

This project transitions from simple early warning detection to a comprehensive 4-phase pipeline that estimates wildfire spread direction directly from monocular camera or UAV imagery, without any external wind-sensor data. The system exploits the physical observation that a smoke plume is a passive tracer of the local wind field.

### Pipeline Overview (4 Phases)
- **Phase A (Detection)**: YOLOv8s (using `yolo26n.pt` from `results/`) object detection for `fire` (cls=0) and `smoke` (cls=1).
- **Phase B (Geometry)**: Geometric direction estimation combining a fire-to-smoke displacement vector (Method A) and PCA smoke-plume axis (Method B).
- **Phase C (Motion)**: Optical-flow confirmation (Farneback dense optical flow) for video inputs, tracking smoke motion.
- **Phase D (Spread Risk)**: Elliptical spread-risk projection following the Rothermel fire-spread model.

### Final Model & System Parameters (from `results/`)
- **Model Weights**: `yolo26n.pt` (to be exported to ONNX).
- **Classes**: `fire`, `smoke`.
- **NMS Thresholds**: Confidence > 0.25, IoU > 0.7.
- **Geometry Conflict Angle (A vs B)**: 60° (flags disagreement without over-triggering on noise).
- **Flow / Geometry Fusion Weight**: 70/30 (Phase C optical flow vs Phase B geometry).
- **Flow / Geometry Conflict Angle**: 45°.
- **EMA Smoothing Factor ($\alpha$)**: 0.3 (unit-circle domain).
- **Motion Floor for Optical Flow**: 50 px.
- **Confidence-Abstention Threshold**: 0.2 (system outputs "undetermined" if below).
- **Spread-Ellipse Geometry**: Radius 15%, Eccentricity 0.8, Offset 40%.

---

## User Review Required

> [!IMPORTANT]
> **Backend Video Processing**: Phase C requires computing dense optical flow between video frames. Processing full videos synchronously via an API might hit timeout limits on Cloud Run (300s). We need to decide if we want to extract frames at a lower FPS (e.g., 5-10 FPS) for the optical flow, or implement an asynchronous task queue.
> **Please confirm if processing videos synchronously at a reduced frame rate is acceptable for this demo.**

> [!WARNING]
> **Model File Mismatch**: The report specifies YOLOv8s, but the weights in the `results/` folder are named `yolo26n.pt` (which indicates a nano model). The pipeline will proceed using `yolo26n.pt` as requested. Please confirm this is the intended weights file.

## Open Questions

1. **Ellipse Rendering**: Should the Rothermel spread ellipse (Phase D) be drawn directly onto the image bytes by the Python backend using OpenCV, or should the backend return the ellipse parameters (center, axes, angle) for the Vite frontend to draw using a Canvas overlay?
2. **Library Dependencies**: To perform PCA and circular EMA smoothing, should we add `scipy` to the backend, or write pure NumPy/OpenCV implementations to keep the Docker image as small as possible?

---

## Proposed Changes

### 0. Environment Setup & Model Optimization

#### [MODIFY] [requirements-export.txt](file:///j:/CPV-AI-PROJEKT/dsp-uav/requirements-export.txt)
Update dependencies if necessary to match the YOLO version required for `yolo26n.pt`.

#### [MODIFY] [scripts/export_onnx.py](file:///j:/CPV-AI-PROJEKT/dsp-uav/scripts/export_onnx.py)
Change the model input path to `results/yolo26n.pt` and output path to `backend/model/best.onnx`.
Ensure `imgsz=640` is used for deployment inference as specified in the report.

### 1. Backend — Flask + ONNX Runtime API Server

#### [MODIFY] [backend/requirements.txt](file:///j:/CPV-AI-PROJEKT/dsp-uav/backend/requirements.txt)
Add `scipy` (if needed for PCA/circular stats) to support Phase B and C logic.

#### [MODIFY] [backend/inference.py](file:///j:/CPV-AI-PROJEKT/dsp-uav/backend/inference.py)
Rewrite the inference engine to implement the 4-phase pipeline:
- **Phase A**: Run ONNX inference, apply NMS. Filter for `fire` and `smoke`.
- **Phase B**:
  - *Method A*: Compute vector from fire centroid to smoke centroid.
  - *Method B*: Create an HSV mask of the smoke box, run PCA to find the dominant eigenvector (elongation axis).
  - *Fusion*: Fuse A and B if both available, checking the 60° conflict angle.
- **Phase C**: For video inputs, compute Farneback optical flow on the smoke mask. Fuse with Phase B at a 70/30 weight. Apply EMA ($\alpha=0.3$).
- **Phase D**: If confidence >= 0.2, generate the Rothermel ellipse (15% radius, 0.8 ecc, 40% offset) rotated by the estimated wind angle. Draw the wind direction arrow (green) and the spread ellipse (orange semi-transparent) on the output frame.

#### [MODIFY] [backend/app.py](file:///j:/CPV-AI-PROJEKT/dsp-uav/backend/app.py)
Update `/api/detect/image` and `/api/detect/video` to return the estimated wind direction angle and confidence score in the JSON response, alongside the annotated media. State management will be needed for `/api/detect/video` to handle the EMA smoothing across frames.

### 2. Frontend — Vite + Tailwind CSS v4 SPA

#### [MODIFY] [frontend/src/pages/landing.js](file:///j:/CPV-AI-PROJEKT/dsp-uav/frontend/src/pages/landing.js)
- Update the abstract to describe the "Wind-Sensor-Free, Physics-Informed Pipeline".
- Update the metrics dashboard (e.g., mAP@50 = 82.6%, 133 FPS).
- Update the gallery to display `fig_spread.png` and `fig_train_and_fps.png` from the `results/` folder.

#### [MODIFY] [frontend/src/pages/detect.js](file:///j:/CPV-AI-PROJEKT/dsp-uav/frontend/src/pages/detect.js)
- Add UI elements to display the overall estimated spread direction (angle) and confidence score.
- Handle cases where the system reports "undetermined" (confidence < 0.2).

---

## Verification Plan

### Automated Tests
1. **Model Export**: Run `export_onnx.py` to ensure `yolo26n.pt` successfully converts to ONNX at 640x640 resolution.
2. **Backend Logic Unit Tests**: Mock YOLO bounding boxes and test Phase B (PCA/Vector) and Phase C (Optical flow) math to ensure angles are calculated correctly and EMA smoothing behaves as expected.

### Integration Test
- Start the Flask backend. Upload an image containing both fire and smoke. Verify the JSON response contains the `direction_angle` and `confidence` fields, and the annotated image contains the green arrow and orange ellipse.
- Upload a short video snippet to verify Phase C (Optical flow) triggers and stabilizes the direction arrow.

### Deployment Verification
- Deploy the updated backend to Cloud Run. Ensure the Docker container size remains manageable even with `scipy`/`opencv-python-headless`.
- Verify the frontend correctly displays the new UI elements on Firebase Hosting.
