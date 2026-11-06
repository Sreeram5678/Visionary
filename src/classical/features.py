"""
Classical Computer Vision: Feature Descriptors.

Includes:
- HSV Color Histograms (paint color distribution invariant to lighting)
- Local Binary Patterns (LBP) texture descriptors (grille, badge, and headlight textures)
- Unified feature vector aggregation for hybrid decision fusion
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.classical.edges import (
    detect_edges_canny,
    extract_contours,
    extract_geometric_features,
)


def extract_color_histogram(
    image: np.ndarray,
    bins: Tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray:
    """
    Extracts a 3D joint color histogram in HSV space and applies L2 normalization.
    HSV separates chromatic content (Hue/Saturation) from intensity (Value).
    
    Args:
        image: RGB input image (uint8 array).
        bins: Number of histogram bins per channel (H, S, V). Default (8, 8, 8) = 512 features.
        
    Returns:
        1D flattened, L2-normalized float32 NumPy array of length (bins[0] * bins[1] * bins[2]).
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Ensure 3-channel RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Compute joint 3D histogram across H [0, 180], S [0, 256], V [0, 256]
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(bins),
        ranges=[0, 180, 0, 256, 0, 256],
    )

    hist = hist.flatten().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(hist)
    if norm > 1e-7:
        hist = hist / norm
    else:
        hist = np.zeros_like(hist)

    return hist


def extract_lbp_texture(
    image: np.ndarray,
    num_points: int = 24,
    radius: int = 3,
    method: str = "uniform",
) -> np.ndarray:
    """
    Extracts Local Binary Patterns (LBP) texture descriptor capturing fine structural
    patterns such as radiator grilles, mesh, badges, and headlight optics.
    
    Args:
        image: RGB or grayscale image (uint8 array).
        num_points: Number of circularly symmetric neighbor set points (P).
        radius: Radius of circle (R).
        method: LBP pattern method ('uniform', 'default', 'ror', 'nri_uniform').
        
    Returns:
        1D L1-normalized float32 NumPy array representing the LBP pattern histogram.
        For 'uniform' method, histogram length is num_points + 2.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to grayscale
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image shape for LBP: {image.shape}")

    # Compute LBP representation
    lbp = local_binary_pattern(gray, P=num_points, R=radius, method=method)

    # Determine number of histogram bins
    if method == "uniform":
        n_bins = int(num_points + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    else:
        n_bins = int(2 ** num_points)
        hist, _ = np.histogram(lbp.ravel(), bins=min(n_bins, 256), density=False)

    hist = hist.astype(np.float32)

    # L1 normalize to form a probability distribution
    hist_sum = hist.sum()
    if hist_sum > 1e-7:
        hist = hist / hist_sum
    else:
        hist = np.zeros_like(hist)

    return hist


def extract_classical_feature_vector(
    image: np.ndarray,
    color_bins: Tuple[int, int, int] = (8, 8, 8),
    lbp_points: int = 24,
    lbp_radius: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Aggregates color histogram, LBP texture, and geometric attributes into a single
    normalized 1D descriptor vector for the hybrid fusion classifier.
    
    Total feature dimensions:
      - Color histogram: 8 * 8 * 8 = 512
      - LBP texture: 24 + 2 = 26
      - Geometric attributes: 6
      - Total = 544 dimensions
      
    Args:
        image: RGB input image (uint8 array).
        color_bins: HSV histogram bins.
        lbp_points: Number of LBP neighbor points.
        lbp_radius: LBP radius.
        
    Returns:
        Tuple of:
          - feature_vector: 1D float32 array (544-dim).
          - geometric_dict: Dictionary containing the raw geometric metrics.
    """
    # 1. Color representation
    color_feat = extract_color_histogram(image, bins=color_bins)

    # 2. Texture representation
    texture_feat = extract_lbp_texture(image, num_points=lbp_points, radius=lbp_radius, method="uniform")

    # 3. Geometric features via edge and contour extraction
    edges = detect_edges_canny(image)
    contours, _ = extract_contours(edges, min_area=50.0)
    geo_metrics = extract_geometric_features(contours, image.shape[:2])

    geo_feat = np.array([
        geo_metrics["aspect_ratio"],
        geo_metrics["compactness"],
        geo_metrics["extent"],
        geo_metrics["solidity"],
        geo_metrics["normalized_perimeter"],
        geo_metrics["normalized_area"],
    ], dtype=np.float32)

    # Concatenate all descriptors
    feature_vector = np.concatenate([color_feat, texture_feat, geo_feat], axis=0)

    return feature_vector, geo_metrics
