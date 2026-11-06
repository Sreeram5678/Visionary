"""
Vehicle Detection & Localization Subsystem for Visionary.

Provides YOLOv8 vehicle detection, spatial padding ROI cropping,
and IoU/mAP benchmark evaluation utilities.
"""

from src.detection.cropper import crop_roi
from src.detection.detector import (
    COCO_VEHICLE_CLASSES,
    DetectionBox,
    DetectionResult,
    VehicleDetector,
)
from src.detection.metrics import (
    compute_batch_iou,
    compute_iou,
    compute_map,
)

__all__ = [
    "COCO_VEHICLE_CLASSES",
    "DetectionBox",
    "DetectionResult",
    "VehicleDetector",
    "crop_roi",
    "compute_iou",
    "compute_batch_iou",
    "compute_map",
]
