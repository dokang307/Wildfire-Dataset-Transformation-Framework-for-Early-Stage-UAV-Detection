"""
ONNX Runtime inference engine for YOLOv8s wildfire detection and direction estimation.
Pure ONNX Runtime + OpenCV + NumPy — no PyTorch dependency at runtime.

The direction-estimation pipeline (Phase B/C/D) is a faithful port of the
validated Kaggle notebook `wildfire-spread-phase03.ipynb`. Angles use the
notebook's **mathematical convention**: degrees measured counter-clockwise from
the +x axis, with +y pointing UP (image y is flipped for all direction math).
"""

import math

import cv2
import numpy as np
import onnxruntime as ort

# Detection classes (index == YOLO class id)
CLASS_NAMES = ["fire", "smoke"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Colors for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "fire": (0, 0, 255),      # Red
    "smoke": (255, 191, 0),   # Deep Sky Blue / Cyan
}

# Default detection thresholds
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU_THRESHOLD = 0.7

# Direction / spread-risk parameters (from notebook & docs/plan.md)
DIRECTION_ABSTAIN_CONF = 0.2   # below this we do not draw the spread overlay
SPREAD_RADIUS = 0.15           # ellipse semi-major = 15% of max(w, h)
SPREAD_ECC = 0.8               # semi-minor = semi-major * (1 - ecc)
SPREAD_OFFSET = 0.4            # ellipse centre offset downwind = 40% of semi-major
EMA_ALPHA = 0.3                # temporal smoothing factor (Phase D)


# ============================================================================
# Phase B/C/D — direction estimation (faithful port of the notebook)
# ============================================================================

def _method_a(fire, smoke):
    """Method A: displacement vector from fire centroid to smoke centroid.

    fire, smoke: normalised (cls_id, cx, cy, w, h). Returns {theta, conf} or None.
    """
    dx, dy = smoke[1] - fire[1], -(smoke[2] - fire[2])  # flip y -> math convention
    dist = math.hypot(dx, dy)
    if dist > 0.02:
        return {"theta": math.degrees(math.atan2(dy, dx)) % 360, "conf": min(dist * 5, 1.0)}
    return None


def _method_b(smoke, img, img_w, img_h):
    """Method B: PCA on the bright/low-saturation smoke pixels inside the box.

    smoke: normalised (cls_id, cx, cy, w, h). Returns {theta, conf} or None.
    """
    if img is None:
        return None
    _, xc, yc, w, h = smoke
    x1, y1 = int((xc - w / 2) * img_w), int((yc - h / 2) * img_h)
    x2, y2 = int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)
    crop = img[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 90))
    if len(xs) <= 200:
        return None
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    eigval, eigvec = np.linalg.eigh(np.cov(pts))
    v = eigvec[:, -1]
    elong = eigval[-1] / max(eigval[0], 1e-6)
    if elong <= 1.5:
        return None
    if v[1] > 0:  # assume plume drifts "up" the image by default
        v = -v
    return {"theta": math.degrees(math.atan2(-v[1], v[0])) % 360, "conf": min((elong - 1) / 4, 1.0)}


def _fuse_ab(ra, rb):
    """Fuse Method A and Method B results into {theta, conf, method} or None."""
    if ra and rb:
        diff = abs((ra["theta"] - rb["theta"] + 180) % 360 - 180)
        best = ra if ra["conf"] >= rb["conf"] else rb
        if diff <= 60:
            return {"theta": best["theta"], "conf": (ra["conf"] + rb["conf"]) / 2, "method": "A+B"}
        return {"theta": best["theta"], "conf": best["conf"] * 0.3, "method": "A+B_conflict"}
    if ra:
        return {"theta": ra["theta"], "conf": ra["conf"] * 0.8, "method": "A"}
    if rb:
        return {"theta": rb["theta"], "conf": rb["conf"] * 0.7, "method": "B"}
    return None


def direction_from_geometry(boxes, img=None, img_w=640, img_h=640):
    """Phase B geometry for the whole frame: the single dominant direction from
    the largest fire and largest smoke (Method A) fused with the largest smoke's
    PCA axis (Method B).

    boxes: list of (cls_id, cx, cy, w, h) with coordinates normalised to [0, 1].
    Returns {'theta', 'conf', 'method'} in math convention, or None.
    """
    fires = [b for b in boxes if b[0] == 0]
    smokes = [b for b in boxes if b[0] == 1]

    ra = rb = None
    if smokes:
        big_smoke = max(smokes, key=lambda b: b[3] * b[4])
        if fires:
            big_fire = max(fires, key=lambda b: b[3] * b[4])
            ra = _method_a(big_fire, big_smoke)
        rb = _method_b(big_smoke, img, img_w, img_h)
    return _fuse_ab(ra, rb)


def directions_from_geometry_multi(boxes, img=None, img_w=640, img_h=640):
    """Phase B geometry per fire: one estimate for every fire, using the smoke
    plume nearest to that fire.

    Returns a list of {'fire', 'theta', 'conf', 'method'}, where 'fire' is the
    normalised box of the fire the estimate belongs to. Empty when no
    fire+smoke pairing yields a direction (matches the single-estimate contract:
    an arrow is only ever anchored on a fire).
    """
    fires = [b for b in boxes if b[0] == 0]
    smokes = [b for b in boxes if b[0] == 1]
    if not fires or not smokes:
        return []

    out = []
    for fire in fires:
        # Nearest smoke by squared centroid distance.
        smoke = min(smokes, key=lambda s: (s[1] - fire[1]) ** 2 + (s[2] - fire[2]) ** 2)
        d = _fuse_ab(_method_a(fire, smoke), _method_b(smoke, img, img_w, img_h))
        if d:
            out.append({"fire": fire, **d})
    return out


def direction_from_flow(prev, curr, boxes, w, h):
    """Phase C: Farneback dense optical flow over the smoke region.

    prev, curr: consecutive BGR frames.
    boxes:      list of (cls_id, cx, cy, w, h) normalised.
    Returns {'theta', 'conf', 'method'} in math convention, or None.
    """
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        if b[0] != 1:  # smoke only
            continue
        _, xc, yc, bw, bh = b
        mask[max(int((yc - bh / 2) * h), 0):int((yc + bh / 2) * h),
             max(int((xc - bw / 2) * w), 0):int((xc + bw / 2) * w)] = True
    if mask.sum() < 100:
        return None

    fx, fy = flow[:, :, 0][mask], flow[:, :, 1][mask]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    strong = mag > np.percentile(mag, 75)
    if strong.sum() < 50:
        return None

    dx, dy = fx[strong].mean(), -fy[strong].mean()  # flip y -> math convention
    return {"theta": math.degrees(math.atan2(dy, dx)) % 360,
            "conf": min(math.hypot(dx, dy) / 5, 1.0), "method": "flow"}


def combine_directions(geom, flow):
    """Fuse Phase B geometry with Phase C flow (70/30 towards flow)."""
    if flow and geom:
        diff = abs((geom["theta"] - flow["theta"] + 180) % 360 - 180)
        if diff <= 45:
            t1, t2 = math.radians(geom["theta"]), math.radians(flow["theta"])
            sx = 0.3 * geom["conf"] * math.cos(t1) + 0.7 * flow["conf"] * math.cos(t2)
            sy = 0.3 * geom["conf"] * math.sin(t1) + 0.7 * flow["conf"] * math.sin(t2)
            return {"theta": math.degrees(math.atan2(sy, sx)) % 360,
                    "conf": min(math.hypot(sx, sy) * 1.2, 1.0), "method": "geom+flow"}
        return {"theta": flow["theta"], "conf": flow["conf"] * 0.7, "method": "flow_override"}
    return flow or geom


class DirectionSmoother:
    """Phase D: exponential moving average on the unit circle."""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha, self.sx, self.sy, self.init = alpha, 0.0, 0.0, False

    def update(self, theta, conf):
        r = math.radians(theta)
        cx, cy = conf * math.cos(r), conf * math.sin(r)
        if not self.init:
            self.sx, self.sy, self.init = cx, cy, True
        else:
            self.sx = self.alpha * cx + (1 - self.alpha) * self.sx
            self.sy = self.alpha * cy + (1 - self.alpha) * self.sy
        return math.degrees(math.atan2(self.sy, self.sx)) % 360, math.hypot(self.sx, self.sy)


class DirectionEstimator:
    """Stateful wrapper around the notebook direction pipeline.

    One instance per media stream: keeps the previous frame (for optical flow)
    and the temporal EMA smoother. For images call estimate(is_video=False);
    for video call estimate(is_video=True) once per processed frame.
    Returns (theta, conf, method); theta/method are None when undetermined.
    """

    def __init__(self, alpha=EMA_ALPHA):
        self.smoother = DirectionSmoother(alpha)
        self.prev_frame = None

    @staticmethod
    def _to_norm_boxes(detections, w, h):
        boxes = []
        for d in detections:
            cid = CLASS_TO_ID.get(d["class"])
            if cid is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            boxes.append((cid, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                          (x2 - x1) / w, (y2 - y1) / h))
        return boxes

    def estimate(self, frame, detections, is_video=False):
        h, w = frame.shape[:2]
        boxes = self._to_norm_boxes(detections, w, h)
        geom = direction_from_geometry(boxes, frame, w, h)

        if not is_video:
            if geom is None:
                return None, 0.0, None
            return geom["theta"], geom["conf"], geom["method"]

        flow = None
        if self.prev_frame is not None:
            flow = direction_from_flow(self.prev_frame, frame, boxes, w, h)
        self.prev_frame = frame.copy()

        d = combine_directions(geom, flow)
        if d is None:
            return None, 0.0, None
        theta, conf = self.smoother.update(d["theta"], d["conf"])
        return theta, conf, d["method"]

    def estimate_multi(self, frame, detections):
        """Per-fire geometry estimates for still images (no flow/EMA).

        Returns a list of {'bbox', 'theta', 'conf', 'method'} — one entry per
        fire that yields a direction — with 'bbox' in pixel [x1, y1, x2, y2].
        Use this to draw an arrow + spread ellipse for every fire in the frame.
        """
        h, w = frame.shape[:2]
        boxes = self._to_norm_boxes(detections, w, h)
        out = []
        for r in directions_from_geometry_multi(boxes, frame, w, h):
            f = r["fire"]
            bbox = [int((f[1] - f[3] / 2) * w), int((f[2] - f[4] / 2) * h),
                    int((f[1] + f[3] / 2) * w), int((f[2] + f[4] / 2) * h)]
            out.append({"bbox": bbox, "theta": r["theta"], "conf": r["conf"], "method": r["method"]})
        return out


# ============================================================================
# Phase A — YOLOv8s detection via ONNX Runtime
# ============================================================================

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

        indices = self._nms(boxes_xyxy, max_scores, class_ids, self.iou_threshold)
        return [{"class": CLASS_NAMES[class_ids[i]], "confidence": float(max_scores[i]),
                 "bbox": [int(x) for x in boxes_xyxy[i]]} for i in indices]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float) -> list[int]:
        max_coordinate = boxes.max()
        offsets = class_ids * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        x1, y1, x2, y2 = boxes_for_nms[:, 0], boxes_for_nms[:, 1], boxes_for_nms[:, 2], boxes_for_nms[:, 3]
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
    def _draw_spread(annotated, bbox, angle, conf, method, big_label):
        """Draw one wind arrow + spread ellipse anchored on a fire box.

        angle is in the math convention (CCW from +x, +y up); it is converted to
        image coordinates here (screen dy = -sin). The ellipse is scaled to the
        fire size (bounded) so several fires can be shown without one ellipse
        swamping the frame.
        """
        h, w = annotated.shape[:2]
        fx1, fy1, fx2, fy2 = bbox
        fcx, fcy = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
        cos_t, sin_t = math.cos(math.radians(angle)), math.sin(math.radians(angle))

        # Semi-major scales with the fire, clamped to [5%, 15%] of max(w, h).
        fire_diag = math.hypot(fx2 - fx1, fy2 - fy1)
        lo, hi = 0.05 * max(w, h), SPREAD_RADIUS * max(w, h)
        major_axis = int(min(max(0.9 * fire_diag, lo), hi))
        minor_axis = max(int(major_axis * (1 - SPREAD_ECC)), 2)
        offset = int(SPREAD_OFFSET * major_axis)

        # Ellipse centre sits downwind of the fire (screen y is flipped).
        center = (int(fcx + offset * cos_t), int(fcy - offset * sin_t))
        ellipse_angle = int(-angle)  # cv2 rotates CW in screen coords

        overlay = annotated.copy()
        cv2.ellipse(overlay, center, (major_axis, minor_axis), ellipse_angle, 0, 360, (0, 140, 255), -1)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        pts = cv2.ellipse2Poly(center, (major_axis, minor_axis), ellipse_angle, 0, 360, 5)
        for i in range(0, len(pts) - 1, 2):
            cv2.line(annotated, tuple(pts[i]), tuple(pts[i + 1]), (0, 140, 255), 2)

        # Wind arrow, length tied to the ellipse so it stays proportional.
        arrow_len = max(major_axis, 40)
        end = (int(fcx + arrow_len * cos_t), int(fcy - arrow_len * sin_t))
        cv2.arrowedLine(annotated, (fcx, fcy), end, (0, 255, 0), 4, tipLength=0.3)

        tag = f" [{method}]" if method else ""
        # cv2 HERSHEY fonts are ASCII-only, so use "deg" not the degree glyph.
        if big_label:
            # Single estimate: centred banner at the top of the frame.
            text = f"wind={int(angle)}deg conf={conf:.2f}{tag}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cx = w // 2
            cv2.rectangle(annotated, (cx - tw // 2 - 10, 0), (cx + tw // 2 + 10, th + 20), (255, 255, 255), -1)
            cv2.putText(annotated, text, (cx - tw // 2, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        else:
            # Multiple estimates: compact label above each fire box.
            text = f"{int(angle)}deg {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(fy1 - 6, th + 4)
            cv2.rectangle(annotated, (fx1, ly - th - 4), (fx1 + tw + 6, ly + 2), (255, 255, 255), -1)
            cv2.putText(annotated, text, (fx1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    @staticmethod
    def draw_detections(img: np.ndarray, detections: list[dict], directions=None) -> np.ndarray:
        """Draw bounding boxes plus one Phase-D spread overlay per direction.

        directions: list of {'theta', 'conf', 'method', 'bbox'(optional pixel
        [x1,y1,x2,y2])}. A direction without 'bbox' is anchored on the largest
        fire (used by the single-estimate video path). Entries with conf below
        the abstain threshold are skipped.
        """
        annotated = img.copy()

        fire_boxes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(det["class"], (0, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            if det["class"] == "fire":
                fire_boxes.append(det["bbox"])

        if not directions:
            return annotated

        # Resolve each direction to a fire anchor and keep the drawable ones.
        drawable = []
        for d in directions:
            angle, conf = d.get("theta"), d.get("conf")
            if angle is None or conf is None or conf < DIRECTION_ABSTAIN_CONF:
                continue
            bbox = d.get("bbox")
            if bbox is None:
                if not fire_boxes:
                    continue
                bbox = max(fire_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            drawable.append((bbox, angle, conf, d.get("method")))

        big_label = len(drawable) == 1
        for bbox, angle, conf, method in drawable:
            OnnxDetector._draw_spread(annotated, bbox, angle, conf, method, big_label)

        return annotated
