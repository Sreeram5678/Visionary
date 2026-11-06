"""
Vehicle Detection & Localization: YOLOv8-based vehicle detector.

Detects cars, trucks, buses, and motorcycles, extracts coordinates,
performs ROI cropping with spatial padding, and annotates images with
bounding boxes and confidence scores.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.detection.cropper import crop_roi

# COCO 80-class vehicle mapping:
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class DetectionBox:
    """Bounding box data structure with class metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        """Computes bounding box area."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class DetectionResult:
    """Comprehensive result from vehicle localization."""
    has_vehicle: bool
    boxes: List[DetectionBox] = field(default_factory=list)
    primary_box: Optional[DetectionBox] = None
    roi: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.uint8))
    annotated_image: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.uint8))
    inference_time_ms: float = 0.0
    status_message: str = "Success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_vehicle": self.has_vehicle,
            "num_detected": len(self.boxes),
            "primary_box": self.primary_box.to_dict() if self.primary_box else None,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "status_message": self.status_message,
        }


class VehicleDetector:
    """
    YOLOv8-based vehicle detector isolating vehicles for fine-grained recognition.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        target_classes: Optional[Dict[int, str]] = None,
    ):
        """
        Initializes the YOLOv8 vehicle detector.
        
        Args:
            weights: Path or name of YOLOv8 model (default 'yolov8n.pt' ~6.2MB).
            conf_threshold: Minimum detection confidence threshold.
            target_classes: Mapping of class_id to label name (defaults to cars/trucks/buses).
        """
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or COCO_VEHICLE_CLASSES
        self._model = None

    def _get_model(self):
        """Lazy loads the YOLO model upon first inference call."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        conf_threshold: Optional[float] = None,
        padding_ratio: float = 0.05,
        target_crop_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> DetectionResult:
        """
        Localizes vehicles in the image, extracts bounding boxes, and crops the primary vehicle ROI.
        
        Args:
            image_input: RGB NumPy array, PIL Image, or file path.
            conf_threshold: Optional override for detection confidence.
            padding_ratio: Spatial margin to expand the ROI bounding box.
            target_crop_size: Optional target size (width, height) for the cropped ROI.
            
        Returns:
            DetectionResult with bounding boxes, cropped vehicle ROI, and visual overlays.
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        rgb_image = self._load_image(image_input)

        if rgb_image.size == 0:
            return DetectionResult(
                has_vehicle=False,
                roi=np.zeros((1, 1, 3), dtype=np.uint8),
                annotated_image=np.zeros((1, 1, 3), dtype=np.uint8),
                status_message="Empty image provided.",
            )

        start_time = time.perf_counter()
        
        try:
            model = self._get_model()
            # Run inference; verbose=False prevents console clutter
            results = model.predict(source=rgb_image, conf=threshold, verbose=False)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return DetectionResult(
                has_vehicle=False,
                roi=rgb_image.copy(),
                annotated_image=rgb_image.copy(),
                inference_time_ms=elapsed_ms,
                status_message=f"Detector inference error: {str(e)}",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        boxes: List[DetectionBox] = []
        if len(results) > 0 and results[0].boxes is not None:
            r_boxes = results[0].boxes
            for i in range(len(r_boxes)):
                cls_id = int(r_boxes.cls[i].item())
                conf = float(r_boxes.conf[i].item())
                
                # Filter strictly to vehicle classes
                if cls_id in self.target_classes and conf >= threshold:
                    coords = r_boxes.xyxy[i].cpu().numpy().tolist()
                    box = DetectionBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.target_classes[cls_id],
                    )
                    boxes.append(box)

        # Sort detected boxes by area descending (primary vehicle usually dominates the scene)
        boxes.sort(key=lambda b: b.area, reverse=True)

        if not boxes:
            return DetectionResult(
                has_vehicle=False,
                boxes=[],
                primary_box=None,
                roi=rgb_image.copy(),
                annotated_image=rgb_image.copy(),
                inference_time_ms=elapsed_ms,
                status_message="No vehicle detected with sufficient confidence.",
            )

        primary = boxes[0]
        roi = crop_roi(
            rgb_image,
            primary.xyxy,
            padding_ratio=padding_ratio,
            target_size=target_crop_size,
        )

        annotated = self._annotate(rgb_image, boxes, primary)

        return DetectionResult(
            has_vehicle=True,
            boxes=boxes,
            primary_box=primary,
            roi=roi,
            annotated_image=annotated,
            inference_time_ms=elapsed_ms,
            status_message="Vehicle successfully localized.",
        )

    def _annotate(
        self,
        image: np.ndarray,
        boxes: List[DetectionBox],
        primary_box: DetectionBox,
    ) -> np.ndarray:
        """Draws bounding boxes and labels on an image."""
        annotated = image.copy()
        
        for box in boxes:
            is_primary = (box == primary_box)
            # Primary vehicle highlighted in bright lime green, secondary vehicles in cyan
            color = (0, 255, 0) if is_primary else (0, 220, 255)
            thickness = 3 if is_primary else 2

            pt1 = (int(round(box.x1)), int(round(box.y1)))
            pt2 = (int(round(box.x2)), int(round(box.y2)))
            cv2.rectangle(annotated, pt1, pt2, color, thickness)

            # Label badge: Class name + Confidence %
            label = f"{box.class_name.upper()} {box.confidence * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            badge_y1 = max(0, pt1[1] - label_h - baseline - 6)
            badge_y2 = pt1[1]
            badge_x1 = pt1[0]
            badge_x2 = pt1[0] + label_w + 10

            # Draw label background badge
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Draw label text in black for high contrast
            cv2.putText(
                annotated,
                label,
                (badge_x1 + 5, badge_y2 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def _load_image(image_input: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Standardizes input to an RGB uint8 NumPy array."""
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = image_input.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return image_input
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                return image_input[:, :, :3]
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                return image_input
            else:
                raise ValueError(f"Unsupported image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
