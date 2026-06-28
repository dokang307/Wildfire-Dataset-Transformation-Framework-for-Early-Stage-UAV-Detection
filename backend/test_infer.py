import cv2
import numpy as np
from inference import OnnxDetector
import sys

def main(image_path):
    print("Loading model...")
    detector = OnnxDetector('model/best.onnx', conf_threshold=0.01)
    
    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image!")
        return

    print("Running detection...")
    detections = detector.detect(img, conf_threshold=0.01)
    
    print(f"Found {len(detections)} detections with threshold 0.01.")
    
    # Let's inspect raw scores without threshold
    input_tensor, ratio, pad = detector._preprocess(img)
    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    predictions = outputs[0][0].T
    class_scores = predictions[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    print("Max score across all predictions:", max_scores.max())
    
    if len(detections) > 0:
        # Sort by conf
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        print("Top 5 detections:")
        for i, d in enumerate(detections[:5]):
            print(f"  {i+1}: {d['class']} {d['confidence']:.4f} {d['bbox']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python test_infer.py <image_path>")
