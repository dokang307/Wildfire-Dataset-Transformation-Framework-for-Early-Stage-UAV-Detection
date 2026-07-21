FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV, plus ffmpeg: opencv-python-headless has no H.264
# encoder, so /api/detect/video falls back to mp4v (which browsers cannot play
# inline) and is re-encoded to H.264 with ffmpeg/libx264.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "--workers", "2", "--threads", "4", "app:app"]
