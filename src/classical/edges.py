"""
Classical Computer Vision: Edge Detection, Contour Extraction, and Geometric Metrics.

Includes:
- Adaptive Canny edge detection (median-based automatic thresholding)
- External and hierarchical contour extraction
- Vehicle geometric descriptors (aspect ratio, compactness, solidity, extent)
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def detect_edges_canny(
    image: np.ndarray,
    sigma: float = 0.33,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
) -> np.ndarray:
    """
    Performs Canny edge detection with automatic thresholding based on image statistics.
    
    If thresholds are not specified, automatic median-based thresholding is applied:
      lower = max(0, (1 - sigma) * median)
      upper = min(255, (1 + sigma) * median)
      
    Args:
        image: Input image (RGB or grayscale uint8 array).
        sigma: Variance parameter controlling threshold range (typically 0.33).
        low_threshold: Optional explicit lower hysteresis threshold.
        high_threshold: Optional explicit upper hysteresis threshold.
        
    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to grayscale if necessary
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image shape for edge detection: {image.shape}")

    # Slight blur to reduce high-frequency salt-and-pepper noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute automatic thresholds if not provided
    if low_threshold is None or high_threshold is None:
        median_val = float(np.median(blurred))
        computed_low = int(max(0, (1.0 - sigma) * median_val))
        computed_high = int(min(255, (1.0 + sigma) * median_val))
        
        # Avoid collapsed thresholds on solid/low-contrast images
        if computed_low == computed_high:
            computed_low = max(0, computed_low - 30)
            computed_high = min(255, computed_high + 30)
            
        low_th = low_threshold if low_threshold is not None else computed_low
        high_th = high_threshold if high_threshold is not None else computed_high
    else:
        low_th = low_threshold
        high_th = high_threshold

    return cv2.Canny(blurred, threshold1=low_th, threshold2=high_th)


def extract_contours(
    edge_map: np.ndarray,
    min_area: float = 100.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extracts significant contours from a binary edge map and generates an annotated visualization.
    
    Args:
        edge_map: Binary edge map (uint8, 0 or 255).
        min_area: Minimum contour area threshold to filter out speckle noise.
        
    Returns:
        Tuple of:
          - filtered_contours: List of contours sorted in descending order of area.
          - visualization: RGB image with contours drawn on a dark background.
    """
    if edge_map is None or edge_map.size == 0:
        raise ValueError("Edge map cannot be None or empty.")

    if edge_map.ndim != 2:
        raise ValueError("Edge map must be a 2D single-channel binary image.")

    # Apply morphological closing to bridge small edge gaps along vehicle body
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny noise contours and sort by area descending
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    filtered.sort(key=cv2.contourArea, reverse=True)

    # Create an RGB contour overlay canvas
    vis = np.zeros((*edge_map.shape, 3), dtype=np.uint8)
    if filtered:
        # Draw all significant contours in cyan
        cv2.drawContours(vis, filtered, -1, (0, 255, 255), 1)
        # Highlight the largest primary contour in bright lime green
        cv2.drawContours(vis, [filtered[0]], -1, (0, 255, 0), 2)

    return filtered, vis


def extract_geometric_features(
    contours: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    """
    Extracts vehicle structural geometric descriptors from the primary contour.
    
    Metrics extracted:
      - aspect_ratio: Bounding box width / height (differentiates sedans vs SUVs vs trucks)
      - compactness: Contour area / (width * height of bounding box)
      - extent: Contour area / bounding box area
      - solidity: Contour area / convex hull area
      - normalized_perimeter: Contour perimeter / sqrt(Image Height * Width)
      - normalized_area: Contour area / (Image Height * Width)
      
    Args:
        contours: List of detected contours (sorted by area descending).
        image_shape: Shape tuple of the image (height, width).
        
    Returns:
        Dictionary of float metric values.
    """
    height, width = image_shape[:2]
    total_area = float(max(1, height * width))
    diag = float(np.sqrt(total_area))

    default_metrics = {
        "aspect_ratio": 1.0,
        "compactness": 0.0,
        "extent": 0.0,
        "solidity": 0.0,
        "normalized_perimeter": 0.0,
        "normalized_area": 0.0,
    }

    if not contours or len(contours) == 0:
        return default_metrics

    primary_contour = contours[0]
    area = float(cv2.contourArea(primary_contour))
    perimeter = float(cv2.arcLength(primary_contour, closed=True))

    x, y, w, h = cv2.boundingRect(primary_contour)
    bbox_area = float(max(1, w * h))

    aspect_ratio = float(w) / float(max(1, h))
    extent = area / bbox_area

    # Convex hull for solidity calculation
    hull = cv2.convexHull(primary_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    compactness = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "aspect_ratio": round(aspect_ratio, 4),
        "compactness": round(compactness, 4),
        "extent": round(extent, 4),
        "solidity": round(solidity, 4),
        "normalized_perimeter": round(perimeter / diag, 4),
        "normalized_area": round(area / total_area, 4),
    }
