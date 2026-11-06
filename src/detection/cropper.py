"""
Vehicle Detection & Localization: Spatial Margin ROI Cropping.

Extracts localized vehicle region with spatial padding (default 5%)
to preserve peripheral elements (spoilers, side mirrors, bumpers)
before passing into the fine-grained deep learning classifier.
"""

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np


def crop_roi(
    image: np.ndarray,
    box: Sequence[float],
    padding_ratio: float = 0.05,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Crops the Region of Interest (ROI) from an image given a bounding box,
    expanding by padding_ratio and clamping within image boundaries.
    
    Args:
        image: Source RGB image (H, W, C) uint8 NumPy array.
        box: Bounding box [x1, y1, x2, y2].
        padding_ratio: Spatial margin expansion ratio on all sides (0.05 = 5%).
        target_size: Optional (width, height) tuple to resize the output crop.
        
    Returns:
        Cropped (and optionally resized) vehicle ROI NumPy array.
    """
    if image is None or image.size == 0:
        raise ValueError("Source image cannot be None or empty.")

    if len(box) != 4:
        raise ValueError("Bounding box must contain exactly 4 coordinates (x1, y1, x2, y2).")

    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]

    # Handle inverted coordinates if any
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Calculate padding margins
    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))

    # Clamp coordinates to image boundaries
    crop_x1 = max(0, int(x_min) - pad_x)
    crop_y1 = max(0, int(y_min) - pad_y)
    crop_x2 = min(img_w, int(x_max) + pad_x)
    crop_y2 = min(img_h, int(y_max) + pad_y)

    # Edge case: degenerate bounding box
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        # Fallback to the whole image
        roi = image.copy()
    else:
        roi = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Optional resize to target dimensions (e.g. 224x224 for ConvNeXt)
    if target_size is not None:
        target_w, target_h = target_size
        roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return roi
