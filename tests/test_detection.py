"""
Unit tests for Vehicle Detection & Localization Subsystem.
Tests IoU calculation, batch IoU, mAP computation, spatial ROI cropping,
and YOLOv8 detector handling of vehicles and edge cases.
"""

import numpy as np
import pytest
from PIL import Image

from src.detection import (
    COCO_VEHICLE_CLASSES,
    DetectionBox,
    DetectionResult,
    VehicleDetector,
    compute_batch_iou,
    compute_iou,
    compute_map,
    crop_roi,
)


# ---------------------------------------------------------------------------
# IoU & mAP Evaluation Metrics Tests
# ---------------------------------------------------------------------------

def test_compute_iou_identical():
    box = [50.0, 50.0, 150.0, 150.0]
    iou = compute_iou(box, box)
    assert pytest.approx(iou, abs=1e-5) == 1.0


def test_compute_iou_disjoint():
    box_a = [0.0, 0.0, 50.0, 50.0]
    box_b = [60.0, 60.0, 100.0, 100.0]
    iou = compute_iou(box_a, box_b)
    assert iou == 0.0


def test_compute_iou_partial():
    # Box A: [0, 0, 10, 10] -> Area 100
    # Box B: [5, 0, 15, 10] -> Area 100
    # Intersection: [5, 0, 10, 10] -> Area 50
    # Union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3 ~ 0.33333
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 0.0, 15.0, 10.0]
    iou = compute_iou(box_a, box_b)
    assert pytest.approx(iou, rel=1e-4) == (1.0 / 3.0)


def test_compute_batch_iou():
    preds = [
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 20.0, 20.0],
    ]
    gts = [
        [0.0, 0.0, 10.0, 10.0],
        [30.0, 30.0, 40.0, 40.0],
    ]
    ious = compute_batch_iou(preds, gts)
    assert len(ious) == 2
    assert pytest.approx(ious[0], abs=1e-5) == 1.0
    assert ious[1] == 0.0


def test_compute_batch_iou_mismatch():
    with pytest.raises(ValueError):
        compute_batch_iou([[0, 0, 1, 1]], [])


def test_compute_map_perfect():
    preds = [[[10, 10, 50, 50, 0.95]]]
    gts = [[[10, 10, 50, 50]]]
    ap = compute_map(preds, gts, iou_threshold=0.5)
    assert pytest.approx(ap, rel=1e-3) == 1.0


def test_compute_map_empty():
    ap = compute_map([], [])
    assert ap == 0.0


# ---------------------------------------------------------------------------
# ROI Cropper Tests
# ---------------------------------------------------------------------------

def test_crop_roi_padding_and_clamping():
    # 200x200 image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = 255  # 100x100 white square

    box = [50.0, 50.0, 150.0, 150.0]  # width 100, height 100
    # 5% padding = 5px on each side -> crop coordinates [45, 45, 155, 155] -> size (110, 110)
    cropped = crop_roi(image, box, padding_ratio=0.05)
    assert cropped.shape == (110, 110, 3)


def test_crop_roi_boundary_clamping():
    # Box located right against top-left edge [0, 0, 50, 50]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = [0.0, 0.0, 50.0, 50.0]
    # Padding cannot go below 0
    cropped = crop_roi(image, box, padding_ratio=0.10)
    # Expected: crop_x1=0, crop_y1=0, crop_x2=55, crop_y2=55
    assert cropped.shape == (55, 55, 3)


def test_crop_roi_target_size():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    box = [40.0, 60.0, 200.0, 180.0]
    cropped = crop_roi(image, box, target_size=(224, 224))
    assert cropped.shape == (224, 224, 3)


def test_crop_roi_invalid():
    with pytest.raises(ValueError):
        crop_roi(None, [0, 0, 10, 10])
    with pytest.raises(ValueError):
        crop_roi(np.zeros((10, 10, 3), dtype=np.uint8), [0, 0])


# ---------------------------------------------------------------------------
# Detector & Data Structure Tests
# ---------------------------------------------------------------------------

def test_detection_box_properties():
    box = DetectionBox(
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=120.0,
        confidence=0.88,
        class_id=2,
        class_name="car",
    )
    assert box.xyxy == (10.0, 20.0, 110.0, 120.0)
    assert box.area == 100.0 * 100.0
    d = box.to_dict()
    assert d["class_name"] == "car"
    assert d["confidence"] == 0.88


def test_vehicle_detector_empty_image():
    detector = VehicleDetector()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    result = detector.detect(empty)
    assert not result.has_vehicle
    assert result.status_message == "Empty image provided."


def test_vehicle_detector_non_car_fallback():
    # Pure black image should result in has_vehicle = False without crashing
    detector = VehicleDetector()
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.detect(black_img)
    
    assert isinstance(result, DetectionResult)
    assert not result.has_vehicle
    assert result.primary_box is None
    assert result.roi.shape == black_img.shape
    assert result.annotated_image.shape == black_img.shape
    assert "No vehicle detected" in result.status_message or "inference" in result.status_message.lower()


def test_vehicle_detector_dict_serialization():
    detector = VehicleDetector()
    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    d = result.to_dict()
    assert "has_vehicle" in d
    assert "num_detected" in d
    assert "inference_time_ms" in d
