import cv2
import numpy as np
from ultralytics import YOLO
from inference import OnnxDetector
import sys

def main(image_path):
    print("=== PyTorch (Ultralytics) ===")
    pt_model = YOLO('../runs/detect/runs/detect/phase1_epoch1-100_20260621_183619/weights/best.pt')
    results = pt_model(image_path, imgsz=1280, conf=0.01)
    pt_boxes = results[0].boxes
    print(f"PT found {len(pt_boxes)} detections.")
    for i in range(min(5, len(pt_boxes))):
        cls_id = int(pt_boxes.cls[i].item())
        conf = pt_boxes.conf[i].item()
        xyxy = pt_boxes.xyxy[i].tolist()
        print(f"  {pt_model.names[cls_id]} {conf:.4f} {xyxy}")

    print("\n=== ONNX Runtime ===")
    detector = OnnxDetector('model/best.onnx', conf_threshold=0.01)
    img = cv2.imread(image_path)
    onnx_dets = detector.detect(img, conf_threshold=0.01)
    print(f"ONNX found {len(onnx_dets)} detections.")
    
    onnx_dets.sort(key=lambda x: x['confidence'], reverse=True)
    for i, d in enumerate(onnx_dets[:5]):
        print(f"  {d['class']} {d['confidence']:.4f} {d['bbox']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python compare_infer.py <image_path>")
