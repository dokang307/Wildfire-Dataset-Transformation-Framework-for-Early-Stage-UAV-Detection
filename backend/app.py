"""
Flask API server for UAV Wildfire Early Detection.
Serves detection endpoints using ONNX Runtime inference.
"""

import base64
import io
import os
import shutil
import subprocess
import tempfile
import time

# Trigger auto-reload for new model
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from inference import OnnxDetector, DirectionEstimator, CLASS_NAMES, DEFAULT_CONFIDENCE

# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "best.onnx"))
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
PORT = int(os.environ.get("PORT", 5000))

# Tried in order; the first two are H.264 (playable inline in browsers), mp4v is
# the always-available fallback so the endpoint never returns an empty file.
VIDEO_CODECS = ("avc1", "H264", "mp4v")
BROWSER_SAFE_CODECS = {"avc1", "H264"}
TRANSCODE_TIMEOUT = 120  # seconds


def _find_ffmpeg():
    """Locate an ffmpeg binary: system PATH first, then the imageio-ffmpeg wheel."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_to_h264(src_path):
    """Re-encode to browser-playable H.264. Returns the new path, or None.

    Many OpenCV builds ship without an H.264 encoder (openh264 is loaded
    dynamically and is often missing), so cv2 falls back to mp4v, which browsers
    cannot play inline. ffmpeg with libx264 fixes that; yuv420p + faststart are
    what make the result actually play in a <video> tag.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None

    dst = tempfile.mktemp(suffix="_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", src_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=TRANSCODE_TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        app.logger.warning("H.264 transcode failed: %s", exc)
        if os.path.exists(dst):
            os.unlink(dst)
        return None

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    return None

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
    return jsonify({"status": "healthy", "model": "YOLOv8s-wildfire-onnx"})


# --- Model Info ---
@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    return jsonify({
        "model": "YOLOv8s",
        "dataset": "FASDD",
        "format": "ONNX",
        "classes": CLASS_NAMES,
        "input_size": f"{detector.input_width}x{detector.input_height}",
        "default_confidence": DEFAULT_CONFIDENCE,
        # Validation metrics (all classes) from results/runs/fasdd_train,
        # completed 50-epoch run (best epoch 50).
        "metrics": {
            "precision": 0.777,
            "recall": 0.695,
            "mAP50": 0.781,
            "mAP50_95": 0.492,
            "f1": 0.734,
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
    
    # Estimate one spread direction per fire
    estimator = DirectionEstimator()
    directions = estimator.estimate_multi(img, detections)

    processing_time = time.time() - start_time

    # Draw bounding boxes and one spread overlay per direction
    annotated = detector.draw_detections(img, detections, directions)

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Representative (highest-confidence) direction for the scalar fields.
    primary = max(directions, key=lambda d: d["conf"], default=None)

    return jsonify({
        "annotated_image": img_base64,
        "detections": detections,
        "direction_angle": primary["theta"] if primary else None,
        "direction_confidence": primary["conf"] if primary else 0.0,
        "direction_method": primary["method"] if primary else None,
        "directions": directions,
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
    transcoded_path = None

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

        # Setup video writer. avc1/H264 is what browsers can play inline, but
        # many OpenCV builds ship without an H.264 encoder, in which case
        # VideoWriter fails to open and silently writes nothing. Fall back so we
        # always produce a file, and fail loudly if no encoder works at all.
        writer, codec = None, None
        for name in VIDEO_CODECS:
            candidate = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*name), fps, (width, height))
            if candidate.isOpened():
                writer, codec = candidate, name
                break
            candidate.release()

        if writer is None:
            return jsonify({
                "error": "No usable video encoder found on the server "
                         f"(tried {', '.join(VIDEO_CODECS)}). Install an H.264-capable OpenCV/FFmpeg build."
            }), 500

        if codec not in BROWSER_SAFE_CODECS:
            app.logger.warning(
                "H.264 unavailable; encoded with '%s'. The file may not play inline in browsers.", codec
            )

        frame_idx = 0
        processed = 0
        last_detections = []
        last_angle = None
        last_conf = 0.0
        last_method = None
        last_directions = []
        estimator = DirectionEstimator()

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Run detection on this frame
                last_detections = detector.detect(frame, conf_threshold=conf)
                last_angle, last_conf, last_method = estimator.estimate(frame, last_detections, is_video=True)
                # Video keeps the temporally-smoothed single global direction
                # (Phase C/D), anchored on the largest fire by draw_detections.
                if last_angle is not None:
                    last_directions = [{"theta": last_angle, "conf": last_conf, "method": last_method}]
                else:
                    last_directions = []
                processed += 1

            # Draw detections (even on skipped frames, use last detections)
            annotated = detector.draw_detections(frame, last_detections, last_directions)
            writer.write(annotated)

            frame_idx += 1

        cap.release()
        writer.release()

        # The writer can still produce nothing (bad codec/size); catch it here
        # instead of letting send_file raise FileNotFoundError.
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return jsonify({"error": f"Video encoding produced no output (codec '{codec}')"}), 500

        # cv2 fell back to a non-browser codec -> re-encode with ffmpeg so the
        # clip plays inline in the frontend's <video> tag.
        send_path = output_path
        if codec not in BROWSER_SAFE_CODECS:
            transcoded_path = _transcode_to_h264(output_path)
            if transcoded_path:
                send_path = transcoded_path
                app.logger.info("Transcoded '%s' output to H.264", codec)
            else:
                app.logger.warning(
                    "Returning '%s' video; install ffmpeg (libx264) for inline browser playback.", codec
                )

        # Read into memory so the temp files can be removed in `finally`.
        with open(send_path, "rb") as fh:
            video_bytes = fh.read()

        return send_file(
            io.BytesIO(video_bytes),
            mimetype="video/mp4",
            as_attachment=True,
            download_name="detected_output.mp4",
        )

    finally:
        # Cleanup temp files
        for path in (input_path, output_path, transcoded_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# --- Error Handlers ---
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
