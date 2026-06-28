"""
Flask API server for UAV Wildfire Early Detection.
Serves detection endpoints using ONNX Runtime inference.
"""

import base64
import io
import os
import tempfile
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from inference import OnnxDetector, CLASS_NAMES, DEFAULT_CONFIDENCE

# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "best.onnx"))
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
PORT = int(os.environ.get("PORT", 5000))

# Allowed origins (update with your Firebase Hosting URL after deployment)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# --- App Setup ---
app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)

# Load model at startup
print(f"Loading ONNX model from: {MODEL_PATH}")
detector = OnnxDetector(MODEL_PATH)
print("Model loaded successfully!")


# --- Health Check ---
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Cloud Run."""
    return jsonify({"status": "healthy", "model": "YOLOv11n-wildfire-onnx"})


# --- Model Info ---
@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    return jsonify({
        "model": "YOLOv11n",
        "format": "ONNX",
        "classes": CLASS_NAMES,
        "input_size": f"{detector.input_width}x{detector.input_height}",
        "default_confidence": DEFAULT_CONFIDENCE,
        "metrics": {
            "precision": 0.9868,
            "recall": 0.9616,
            "mAP50": 0.9797,
            "mAP50_95": 0.8911,
            "f1": 0.97,
        },
    })


# --- Image Detection ---
@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    """
    Detect wildfire in an uploaded image.

    Expects multipart form with:
        - file: image file (jpg, jpeg, png)
        - confidence: (optional) confidence threshold (0.1-0.9)

    Returns JSON with:
        - annotated_image: base64-encoded annotated image
        - detections: list of detection objects
        - processing_time: inference time in seconds
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Validate file type
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Invalid file type: {ext}. Allowed: {allowed_ext}"}), 400

    # Read image
    file_bytes = file.read()
    if len(file_bytes) > MAX_IMAGE_SIZE:
        return jsonify({"error": f"File too large. Max: {MAX_IMAGE_SIZE // (1024*1024)} MB"}), 400

    # Decode image
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Get confidence threshold
    conf = request.form.get("confidence", DEFAULT_CONFIDENCE)
    try:
        conf = float(conf)
        conf = max(0.01, min(0.9, conf))
    except (ValueError, TypeError):
        conf = DEFAULT_CONFIDENCE

    # Run detection
    start_time = time.time()
    detections = detector.detect(img, conf_threshold=conf)
    processing_time = time.time() - start_time

    # Draw bounding boxes
    annotated = detector.draw_detections(img, detections)

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "annotated_image": img_base64,
        "detections": detections,
        "processing_time": round(processing_time, 3),
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "confidence_threshold": conf,
    })


# --- Video Detection ---
@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    """
    Detect wildfire in an uploaded video.

    Expects multipart form with:
        - file: video file (mp4)
        - confidence: (optional) confidence threshold
        - frame_skip: (optional) process every Nth frame (default 1)

    Returns: annotated MP4 video file.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        return jsonify({"error": "Invalid video format. Allowed: mp4, avi, mov"}), 400

    # Get parameters
    conf = request.form.get("confidence", DEFAULT_CONFIDENCE)
    try:
        conf = float(conf)
        conf = max(0.01, min(0.9, conf))
    except (ValueError, TypeError):
        conf = DEFAULT_CONFIDENCE

    frame_skip_str = request.form.get("frame_skip")
    try:
        frame_skip = int(frame_skip_str)
    except (ValueError, TypeError):
        frame_skip = None  # Will be calculated based on FPS

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        file.save(tmp_in)
        input_path = tmp_in.name

    # Output temp file
    output_path = tempfile.mktemp(suffix=".mp4")

    try:
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if frame_skip is None:
            # Process max 5 frames per second to massively speed up video processing
            frame_skip = max(1, int(fps / 5))
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit video length (max ~30 seconds of processed frames)
        max_frames = int(30 * fps)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        last_detections = []

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Run detection on this frame
                last_detections = detector.detect(frame, conf_threshold=conf)
                processed += 1

            # Draw detections (even on skipped frames, use last detections)
            annotated = detector.draw_detections(frame, last_detections)
            writer.write(annotated)

            frame_idx += 1

        cap.release()
        writer.release()

        # Send the annotated video
        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="detected_output.mp4",
        )

    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.unlink(input_path)
        # Note: output_path cleanup happens after send_file completes


# --- Error Handlers ---
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
