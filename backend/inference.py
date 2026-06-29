"""
ONNX Runtime inference engine for YOLOv8s-P2 wildfire detection.
Pure ONNX Runtime — no PyTorch dependency at runtime.
"""

import cv2
import numpy as np
import onnxruntime as ort


# Detection classes
CLASS_NAMES = ["Early_Fire", "Early_Smoke"]

# Colors for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "Early_Fire": (0, 0, 220),      # Red
    "Early_Smoke": (0, 140, 255),    # Orange
}

# Default confidence threshold (optimal F1 from training)
DEFAULT_CONFIDENCE = 0.38
DEFAULT_IOU_THRESHOLD = 0.7


class OnnxDetector:
    """YOLOv8s-P2 object detector using ONNX Runtime."""

    def __init__(self, model_path: str, conf_threshold: float = DEFAULT_CONFIDENCE,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        """
        Initialize the ONNX detector.

        Args:
            model_path: Path to the .onnx model file.
            conf_threshold: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for non-max suppression.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Create ONNX Runtime session
        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

        # Get model input details
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        input_shape = model_input.shape
        # Handle dynamic dimensions
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 1280
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 1280

        print(f"[OnnxDetector] Model loaded: {model_path}")
        print(f"[OnnxDetector] Input: {self.input_name} shape={input_shape}")
        print(f"[OnnxDetector] Using size: {self.input_width}x{self.input_height}")

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize image with letterboxing (maintain aspect ratio with padding).

        Returns:
            (letterboxed_image, scale_ratio, (pad_w, pad_h))
        """
        shape = img.shape[:2]  # (h, w)
        new_shape = (self.input_height, self.input_width)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute new unpadded dimensions
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # Compute padding
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, r, (int(round(dw)), int(round(dh)))

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess image for YOLO inference.

        Args:
            img: BGR image (OpenCV format).

        Returns:
            (input_tensor, scale_ratio, (pad_w, pad_h))
        """
        # Letterbox resize
        letterboxed, ratio, (pad_w, pad_h) = self._letterbox(img)

        # BGR → RGB
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC → CHW → NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched, ratio, (pad_w, pad_h)

    def _postprocess(self, output: np.ndarray, ratio: float,
                     pad: tuple[int, int], orig_shape: tuple[int, int]) -> list[dict]:
        """
        Postprocess YOLO output: extract detections with NMS.

        Args:
            output: Raw model output, shape (1, num_classes+4, num_predictions).
            ratio: Scale ratio from letterboxing.
            pad: (pad_w, pad_h) from letterboxing.
            orig_shape: Original image (h, w).

        Returns:
            List of detection dicts with keys: class, confidence, bbox (x1, y1, x2, y2).
        """
        # output shape: (1, 4+num_classes, num_preds) → transpose to (num_preds, 4+num_classes)
        predictions = output[0].T  # (num_preds, 6) for 2 classes

        # Extract boxes (cx, cy, w, h) and class scores
        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Get max class score and class id for each prediction
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores >= self.conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, x2, y2
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

        # Remove padding and rescale to original image coordinates
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad[0]) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad[1]) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad[0]) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad[1]) / ratio

        # Clip to image boundaries
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_shape[1])
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_shape[0])
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_shape[1])
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_shape[0])

        # Apply NMS
        indices = self._nms(boxes_xyxy, max_scores, self.iou_threshold)

        detections = []
        for i in indices:
            detections.append({
                "class": CLASS_NAMES[class_ids[i]],
                "confidence": float(max_scores[i]),
                "bbox": [
                    int(boxes_xyxy[i, 0]),
                    int(boxes_xyxy[i, 1]),
                    int(boxes_xyxy[i, 2]),
                    int(boxes_xyxy[i, 3]),
                ],
            })

        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray, conf_threshold: float = None) -> list[dict]:
        """
        Run detection on a single image.

        Args:
            img: BGR image (OpenCV format).
            conf_threshold: Override confidence threshold for this call.

        Returns:
            List of detection dicts.
        """
        if conf_threshold is not None:
            old_conf = self.conf_threshold
            self.conf_threshold = conf_threshold

        orig_shape = img.shape[:2]  # (h, w)

        # Preprocess
        input_tensor, ratio, pad = self._preprocess(img)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess
        detections = self._postprocess(outputs[0], ratio, pad, orig_shape)

        if conf_threshold is not None:
            self.conf_threshold = old_conf

        return detections

    @staticmethod
    def draw_detections(img: np.ndarray, detections: list[dict],
                        line_width: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.

        Args:
            img: BGR image to draw on (will be copied).
            detections: List of detection dicts from detect().
            line_width: Bounding box line thickness.

        Returns:
            Annotated image copy.
        """
        annotated = img.copy()

        for det in detections:
            cls_name = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        return annotated
