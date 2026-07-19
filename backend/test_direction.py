"""
Unit tests for the direction-estimation pipeline (Phase B/C/D).

These assert the *math convention* of the notebook port: angles are measured
counter-clockwise from +x with +y pointing UP, so:
    east=0deg, north=90deg, west=180deg, south=270deg.
Detections use pixel bboxes [x1, y1, x2, y2] (image y grows downward).
"""

import math
import unittest

import cv2
import numpy as np

from inference import (
    DirectionEstimator,
    DirectionSmoother,
    OnnxDetector,
    combine_directions,
    direction_from_geometry,
    directions_from_geometry_multi,
)


def _box(cls_id, cx, cy, w=0.1, h=0.1):
    """Build a normalised (cls_id, cx, cy, w, h) tuple."""
    return (cls_id, cx, cy, w, h)


def _det(cls_name, cx, cy, w, h, img_w=640, img_h=640):
    """Build a detection dict with a pixel bbox centred at (cx, cy) normalised."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return {"class": cls_name, "confidence": 0.9, "bbox": [x1, y1, x2, y2]}


def _angle_close(a, b, tol=1.0):
    diff = abs((a - b + 180) % 360 - 180)
    return diff <= tol


class TestGeometryMethodA(unittest.TestCase):
    """Method A: fire->smoke displacement vector."""

    def test_smoke_above_fire_points_north(self):
        # fire low on the image (cy=0.7), smoke high (cy=0.3) -> plume drifts up
        boxes = [_box(0, 0.5, 0.7), _box(1, 0.5, 0.3, 0.2, 0.2)]
        r = direction_from_geometry(boxes, img=None)
        self.assertIsNotNone(r)
        self.assertEqual(r["method"], "A")
        self.assertTrue(_angle_close(r["theta"], 90), r["theta"])

    def test_smoke_right_of_fire_points_east(self):
        boxes = [_box(0, 0.3, 0.5), _box(1, 0.7, 0.5, 0.2, 0.2)]
        r = direction_from_geometry(boxes, img=None)
        self.assertTrue(_angle_close(r["theta"], 0), r["theta"])

    def test_smoke_left_of_fire_points_west(self):
        boxes = [_box(0, 0.7, 0.5), _box(1, 0.3, 0.5, 0.2, 0.2)]
        r = direction_from_geometry(boxes, img=None)
        self.assertTrue(_angle_close(r["theta"], 180), r["theta"])

    def test_conf_scales_with_distance(self):
        near = direction_from_geometry([_box(0, 0.5, 0.55), _box(1, 0.5, 0.45)], img=None)
        far = direction_from_geometry([_box(0, 0.5, 0.9), _box(1, 0.5, 0.1)], img=None)
        self.assertLess(near["conf"], far["conf"])

    def test_tiny_displacement_abstains(self):
        # displacement < 0.02 and no image for Method B -> undetermined
        boxes = [_box(0, 0.5, 0.5), _box(1, 0.505, 0.5)]
        self.assertIsNone(direction_from_geometry(boxes, img=None))

    def test_no_boxes_abstains(self):
        self.assertIsNone(direction_from_geometry([], img=None))


class TestGeometryMethodB(unittest.TestCase):
    """Method B: PCA on the smoke plume shape."""

    def test_diagonal_plume_axis(self):
        # A thick white streak running bottom-left -> top-right (up-right on screen)
        # is a plume drifting toward the north-east -> ~45deg.
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(img, (30, 170), (170, 30), (255, 255, 255), 25)
        boxes = [_box(1, 0.5, 0.5, 1.0, 1.0)]  # smoke box = whole frame
        r = direction_from_geometry(boxes, img=img, img_w=200, img_h=200)
        self.assertIsNotNone(r)
        self.assertEqual(r["method"], "B")
        self.assertTrue(_angle_close(r["theta"], 45, tol=15), r["theta"])


class TestCombine(unittest.TestCase):
    def test_geom_only_passthrough(self):
        geom = {"theta": 90.0, "conf": 0.7, "method": "A"}
        self.assertEqual(combine_directions(geom, None), geom)

    def test_flow_only_passthrough(self):
        flow = {"theta": 10.0, "conf": 0.5, "method": "flow"}
        self.assertEqual(combine_directions(None, flow), flow)

    def test_aligned_fusion(self):
        geom = {"theta": 90.0, "conf": 0.8, "method": "A+B"}
        flow = {"theta": 100.0, "conf": 0.8, "method": "flow"}
        out = combine_directions(geom, flow)
        self.assertEqual(out["method"], "geom+flow")
        self.assertTrue(90 <= out["theta"] <= 100, out["theta"])

    def test_conflict_prefers_flow(self):
        geom = {"theta": 0.0, "conf": 0.8, "method": "A+B"}
        flow = {"theta": 170.0, "conf": 0.8, "method": "flow"}
        out = combine_directions(geom, flow)
        self.assertEqual(out["method"], "flow_override")
        self.assertTrue(_angle_close(out["theta"], 170))


class TestSmoother(unittest.TestCase):
    def test_first_update_is_identity(self):
        s = DirectionSmoother(alpha=0.3)
        theta, conf = s.update(90.0, 1.0)
        self.assertTrue(_angle_close(theta, 90))
        self.assertAlmostEqual(conf, 1.0, places=3)

    def test_converges_toward_stable_signal(self):
        s = DirectionSmoother(alpha=0.3)
        s.update(0.0, 1.0)
        for _ in range(20):
            theta, _ = s.update(90.0, 1.0)
        self.assertTrue(_angle_close(theta, 90, tol=2), theta)

    def test_wraparound_mean(self):
        # 350deg and 10deg should average near 0deg, not 180deg
        s = DirectionSmoother(alpha=0.5)
        s.update(350.0, 1.0)
        theta, _ = s.update(10.0, 1.0)
        self.assertTrue(_angle_close(theta, 0, tol=2), theta)


class TestEstimatorImageMode(unittest.TestCase):
    """DirectionEstimator end-to-end on pixel detections (no EMA for images)."""

    def test_image_mode_uses_geometry(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = [_det("fire", 0.5, 0.7, 0.1, 0.1), _det("smoke", 0.5, 0.3, 0.2, 0.2)]
        est = DirectionEstimator()
        theta, conf, method = est.estimate(frame, dets, is_video=False)
        self.assertTrue(_angle_close(theta, 90), theta)
        self.assertEqual(method, "A")
        self.assertGreater(conf, 0.2)

    def test_image_mode_no_detections_abstains(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        theta, conf, method = DirectionEstimator().estimate(frame, [], is_video=False)
        self.assertIsNone(theta)
        self.assertIsNone(method)
        self.assertEqual(conf, 0.0)


class TestMultiDirection(unittest.TestCase):
    """Per-fire estimates for images with several fires/smokes."""

    def test_one_estimate_per_fire(self):
        # Two fires, one shared smoke above them -> two estimates, both ~north.
        boxes = [
            _box(0, 0.3, 0.7), _box(0, 0.7, 0.7),           # two fires, low
            _box(1, 0.5, 0.3, 0.4, 0.3),                    # one wide smoke, high
        ]
        out = directions_from_geometry_multi(boxes, img=None)
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertIn("fire", r)
            self.assertTrue(60 <= r["theta"] <= 120, r["theta"])

    def test_nearest_smoke_pairing(self):
        # Fire-left pairs with smoke-left (points west->... actually up-left),
        # fire-right pairs with smoke-right. The two thetas must differ.
        boxes = [
            _box(0, 0.2, 0.6), _box(0, 0.8, 0.6),
            _box(1, 0.1, 0.3), _box(1, 0.9, 0.3),
        ]
        out = directions_from_geometry_multi(boxes, img=None)
        self.assertEqual(len(out), 2)
        thetas = sorted(r["theta"] for r in out)
        self.assertGreater(abs(thetas[0] - thetas[1]), 10)

    def test_no_smoke_gives_empty(self):
        self.assertEqual(directions_from_geometry_multi([_box(0, 0.5, 0.5)], img=None), [])

    def test_estimate_multi_returns_pixel_bbox(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = [_det("fire", 0.3, 0.7, 0.1, 0.1), _det("fire", 0.7, 0.7, 0.1, 0.1),
                _det("smoke", 0.5, 0.3, 0.4, 0.2)]
        out = DirectionEstimator().estimate_multi(frame, dets)
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertEqual(len(r["bbox"]), 4)
            self.assertTrue(all(isinstance(v, int) for v in r["bbox"]))

    def test_draw_multiple_overlays(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            {"class": "fire", "confidence": 0.9, "bbox": [180, 300, 240, 360]},
            {"class": "fire", "confidence": 0.9, "bbox": [420, 300, 480, 360]},
            {"class": "smoke", "confidence": 0.9, "bbox": [150, 60, 500, 200]},
        ]
        directions = DirectionEstimator().estimate_multi(frame, dets)
        self.assertEqual(len(directions), 2)
        out = OnnxDetector.draw_detections(frame, dets, directions)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int((out != frame).sum()), 0)

    def test_draw_empty_directions_only_boxes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"class": "fire", "confidence": 0.9, "bbox": [180, 300, 240, 360]}]
        out = OnnxDetector.draw_detections(frame, dets, [])
        # boxes drawn, but no arrow/ellipse
        self.assertGreater(int((out != frame).sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
