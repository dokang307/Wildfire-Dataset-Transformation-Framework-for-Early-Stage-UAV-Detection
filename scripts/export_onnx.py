"""
Export YOLOv11n best.pt to ONNX format for optimized inference.

Run this script inside the export virtual environment:
    .\.venv\Scripts\Activate.ps1
    python scripts/export_onnx.py
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    weights_path = (
        project_root
        / "runs"
        / "detect"
        / "runs"
        / "detect"
        / "phase1_epoch1-100_20260621_183619"
        / "weights"
        / "best.pt"
    )
    output_dir = project_root / "backend" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    # Export to ONNX
    print("Exporting to ONNX format...")
    export_path = model.export(
        format="onnx",
        imgsz=1280,
        simplify=True,
        dynamic=True,
        opset=17,
    )

    # Move exported ONNX to backend/model/
    export_path = Path(export_path)
    dest_path = output_dir / "best.onnx"

    if export_path != dest_path:
        shutil.move(str(export_path), str(dest_path))

    print(f"✅ ONNX model exported to: {dest_path}")
    print(f"   File size: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
