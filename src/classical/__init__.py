"""
Classical Computer Vision Subsystem for Visionary.

Provides illumination normalization, denoising, edge and contour detection,
and texture/color descriptors.
"""

from src.classical.edges import (
    detect_edges_canny,
    extract_contours,
    extract_geometric_features,
)
from src.classical.features import (
    extract_classical_feature_vector,
    extract_color_histogram,
    extract_lbp_texture,
)
from src.classical.pipeline import (
    ClassicalPipeline,
    ClassicalResult,
)
from src.classical.preprocessing import (
    apply_clahe,
    denoise_bilateral,
    gaussian_blur,
)

__all__ = [
    "apply_clahe",
    "denoise_bilateral",
    "gaussian_blur",
    "detect_edges_canny",
    "extract_contours",
    "extract_geometric_features",
    "extract_color_histogram",
    "extract_lbp_texture",
    "extract_classical_feature_vector",
    "ClassicalPipeline",
    "ClassicalResult",
]
