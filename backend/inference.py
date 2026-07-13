"""
ONNX Runtime inference engine for YOLOv8s wildfire detection and direction estimation.
Pure ONNX Runtime + OpenCV — no PyTorch dependency at runtime.
"""

import cv2
import numpy as np
import onnxruntime as ort
import math

# Detection classes
CLASS_NAMES = ["fire", "smoke"]

# Colors for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "fire": (0, 0, 220),      # Red
    "smoke": (0, 140, 255),   # Orange
}

# Default confidence threshold
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.7


def _angle_diff(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def _circular_mean(angles, weights=None):
    if weights is None:
        weights = [1.0] * len(angles)
    sin_sum = sum(w * math.sin(math.radians(a)) for a, w in zip(angles, weights))
    cos_sum = sum(w * math.cos(math.radians(a)) for a, w in zip(angles, weights))
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360


class DirectionEstimator:
    def __init__(self):
        self.prev_frame_gray = None
        self.ema_angle = None
        self.ema_conf = None
        self.alpha = 0.3

    def estimate(self, frame, detections, is_video=False):
        fire_boxes = [d['bbox'] for d in detections if d['class'] == 'fire']
        smoke_boxes = [d['bbox'] for d in detections if d['class'] == 'smoke']
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not smoke_boxes:
            self.prev_frame_gray = curr_gray
            # Decay confidence if no smoke
            if self.ema_conf is not None:
                self.ema_conf *= 0.8
            return self.ema_angle, self.ema_conf if self.ema_conf else 0.0
            
        smoke_box = max(smoke_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        sx1, sy1, sx2, sy2 = smoke_box
        
        method_a_angle = None
        method_b_angle = None
        
        # Method A: Fire to Smoke vector
        if fire_boxes:
            fire_box = max(fire_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            fx1, fy1, fx2, fy2 = fire_box
            fcx, fcy = (fx1+fx2)/2.0, (fy1+fy2)/2.0
            scx, scy = (sx1+sx2)/2.0, (sy1+sy2)/2.0
            dx, dy = scx - fcx, scy - fcy
            # Only use if distance is significant
            if math.hypot(dx, dy) > 10:
                method_a_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            
        # Method B: PCA on Smoke HSV
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 100])
            upper = np.array([180, 80, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 20:
                pts = np.vstack((x_coords, y_coords)).astype(np.float64).T
                mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                if eigenvectors is not None and len(eigenvectors) > 0:
                    ex, ey = eigenvectors[0]
                    angle = (math.degrees(math.atan2(ey, ex)) + 360) % 360
                    # Disambiguate
                    if method_a_angle is not None:
                        if _angle_diff(angle, method_a_angle) > 90:
                            angle = (angle + 180) % 360
                    else:
                        if ey > 0: # Assume smoke goes up if no fire box
                            angle = (angle + 180) % 360
                    method_b_angle = angle

        # Fuse A and B
        geom_angle = None
        geom_conf = 0.0
        if method_a_angle is not None and method_b_angle is not None:
            if _angle_diff(method_a_angle, method_b_angle) <= 60:
                geom_angle = _circular_mean([method_a_angle, method_b_angle])
                geom_conf = 1.0
        elif method_a_angle is not None:
            geom_angle = method_a_angle
            geom_conf = 0.7
        elif method_b_angle is not None:
            geom_angle = method_b_angle
            geom_conf = 0.7
            
        # Phase C: Optical Flow (only if video)
        flow_angle = None
        if is_video and self.prev_frame_gray is not None and roi.size > 0:
            prev_roi = self.prev_frame_gray[sy1:sy2, sx1:sx2]
            curr_roi = curr_gray[sy1:sy2, sx1:sx2]
            if prev_roi.shape == curr_roi.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0,0,100]), np.array([180,80,255]))
                flow_x = flow[:,:,0][mask > 0]
                flow_y = flow[:,:,1][mask > 0]
                if len(flow_x) > 0:
                    med_x = np.median(flow_x)
                    med_y = np.median(flow_y)
                    magnitude = math.sqrt(med_x**2 + med_y**2)
                    if magnitude > 1.0: # 50px was over 0.5s, for sequential frames use a smaller floor
                        flow_angle = (math.degrees(math.atan2(med_y, med_x)) + 360) % 360
        
        self.prev_frame_gray = curr_gray
        
        # Fuse Geometry and Flow
        final_angle = None
        final_conf = 0.0
        
        if is_video:
            if geom_angle is not None and flow_angle is not None:
                if _angle_diff(geom_angle, flow_angle) <= 45:
                    final_angle = _circular_mean([geom_angle, flow_angle], [0.3, 0.7])
                    final_conf = min(1.0, geom_conf + 0.2)
                else:
                    final_angle = flow_angle
                    final_conf = 0.8
            elif geom_angle is not None:
                final_angle = geom_angle
                final_conf = geom_conf
            elif flow_angle is not None:
                final_angle = flow_angle
                final_conf = 0.8
        else:
            final_angle = geom_angle
            final_conf = geom_conf
            
        # Phase D: EMA Smoothing
        if final_angle is not None:
            if self.ema_angle is None or not is_video:
                self.ema_angle = final_angle
                self.ema_conf = final_conf
            else:
                self.ema_angle = _circular_mean([self.ema_angle, final_angle], [1-self.alpha, self.alpha])
                self.ema_conf = (1 - self.alpha) * self.ema_conf + self.alpha * final_conf
                
        return self.ema_angle, self.ema_conf if self.ema_conf else 0.0


class OnnxDetector:
    """YOLOv8s object detector using ONNX Runtime."""

    def __init__(self, model_path: str, conf_threshold: float = DEFAULT_CONFIDENCE,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        input_shape = model_input.shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        shape = img.shape[:2]
        new_shape = (self.input_height, self.input_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (int(round(dw)), int(round(dh)))

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        letterboxed, ratio, pad = self._letterbox(img)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        batched = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
        return batched, ratio, pad

    def _postprocess(self, output: np.ndarray, ratio: float, pad: tuple[int, int], orig_shape: tuple[int, int]) -> list[dict]:
        predictions = output[0].T
        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores >= self.conf_threshold
        boxes_cxcywh, max_scores, class_ids = boxes_cxcywh[mask], max_scores[mask], class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad[0]) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad[1]) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad[0]) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad[1]) / ratio

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])

        indices = self._nms(boxes_xyxy, max_scores, self.iou_threshold)
        return [{"class": CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else f"class_{class_ids[i]}", "confidence": float(max_scores[i]),
                 "bbox": [int(x) for x in boxes_xyxy[i]]} for i in indices]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
        old_conf = self.conf_threshold
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            
        orig_shape = img.shape[:2]
        input_tensor, ratio, pad = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self._postprocess(outputs[0], ratio, pad, orig_shape)
        
        self.conf_threshold = old_conf
        return detections

    @staticmethod
    def draw_detections(img: np.ndarray, detections: list[dict], angle: float = None, conf: float = None) -> np.ndarray:
        annotated = img.copy()
        
        # Bounding boxes
        fire_boxes = []
        for det in detections:
            cls_name = det["class"]
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {det['confidence']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if cls_name == "fire":
                fire_boxes.append(det["bbox"])
                
        # Phase D: Spread Risk overlay
        if angle is not None and conf is not None and conf >= 0.2:
            if fire_boxes:
                # Use largest fire box as origin
                fire_box = max(fire_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                fx1, fy1, fx2, fy2 = fire_box
                fcx, fcy = int((fx1+fx2)/2), int((fy1+fy2)/2)
                
                h, w = img.shape[:2]
                major_axis = int(0.15 * w)
                minor_axis = int(major_axis * math.sqrt(1 - 0.8**2))
                offset = int(0.40 * major_axis)
                
                center_x = int(fcx + offset * math.cos(math.radians(angle)))
                center_y = int(fcy + offset * math.sin(math.radians(angle)))
                
                # Draw arrow
                end_x = int(fcx + 80 * math.cos(math.radians(angle)))
                end_y = int(fcy + 80 * math.sin(math.radians(angle)))
                cv2.arrowedLine(annotated, (fcx, fcy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
                
                # Draw ellipse (transparent)
                overlay = annotated.copy()
                cv2.ellipse(overlay, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, (0, 140, 255), -1)
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                
        return annotated
