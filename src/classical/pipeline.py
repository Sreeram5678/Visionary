"""
Classical Computer Vision: Unified Pipeline Orchestrator.

Combines illumination normalization, denoising, edge analysis, contour extraction,
and feature descriptor computation into a single reproducible pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

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
from src.classical.preprocessing import (
    apply_clahe,
    denoise_bilateral,
    gaussian_blur,
)


@dataclass
class ClassicalResult:
    """Structured container for classical CV outputs."""
    features: np.ndarray
    metrics: Dict[str, float]
    visualizations: Dict[str, np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features_dim": int(self.features.shape[0]),
            "metrics": self.metrics,
            "available_visualizations": list(self.visualizations.keys()),
        }


class ClassicalPipeline:
    """
    End-to-end Classical Computer Vision Processing Pipeline for vehicle images.
    
    Stages:
      1. Image Ingestion & RGB standardisation
      2. Illumination Normalization via CLAHE (in CIE LAB space)
      3. Edge-preserving Bilateral Denoising
      4. Adaptive Canny Edge Detection & Contours
      5. Descriptor Extraction (Color Histogram, LBP Texture, Geometric Ratios)
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        canny_sigma: float = 0.33,
        min_contour_area: float = 50.0,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.canny_sigma = canny_sigma
        self.min_contour_area = min_contour_area

    def process(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
    ) -> ClassicalResult:
        """
        Runs the full classical CV pipeline on the input image.
        
        Args:
            image_input: Image path, PIL Image, or NumPy array (RGB uint8).
            
        Returns:
            ClassicalResult containing feature vector, metrics, and visualization frames.
        """
        # 1. Normalize input to RGB NumPy array
        rgb_image = self._load_image(image_input)

        # 2. Illumination Normalization (CLAHE)
        clahe_image = apply_clahe(
            rgb_image,
            clip_limit=self.clahe_clip_limit,
            tile_grid_size=self.clahe_tile_grid,
        )

        # 3. Edge-Preserving Denoising
        denoised_image = denoise_bilateral(
            clahe_image,
            d=self.bilateral_d,
            sigma_color=self.bilateral_sigma_color,
            sigma_space=self.bilateral_sigma_space,
        )

        # 4. Adaptive Canny Edge Detection
        edge_map = detect_edges_canny(
            denoised_image,
            sigma=self.canny_sigma,
        )

        # 5. Contour Extraction & Visualization
        contours, contour_overlay = extract_contours(
            edge_map,
            min_area=self.min_contour_area,
        )

        # 6. Geometric Features
        geometric_metrics = extract_geometric_features(
            contours,
            rgb_image.shape[:2],
        )

        # 7. Color (HSV) & Texture (LBP) Descriptors
        color_hist = extract_color_histogram(clahe_image)
        lbp_hist = extract_lbp_texture(clahe_image)
        
        geo_vector = np.array([
            geometric_metrics["aspect_ratio"],
            geometric_metrics["compactness"],
            geometric_metrics["extent"],
            geometric_metrics["solidity"],
            geometric_metrics["normalized_perimeter"],
            geometric_metrics["normalized_area"],
        ], dtype=np.float32)

        # Combined 544-dimensional feature vector
        combined_features = np.concatenate([color_hist, lbp_hist, geo_vector], axis=0)

        # Create 3-channel version of edge map for easy UI display
        edges_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)

        visualizations = {
            "raw": rgb_image,
            "clahe": clahe_image,
            "denoised": denoised_image,
            "edges": edges_rgb,
            "contours": contour_overlay,
        }

        return ClassicalResult(
            features=combined_features,
            metrics=geometric_metrics,
            visualizations=visualizations,
        )

    @staticmethod
    def _load_image(image_input: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Helper to ensure input is a valid RGB uint8 NumPy array."""
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image from path: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = image_input.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Image array cannot be empty.")
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
