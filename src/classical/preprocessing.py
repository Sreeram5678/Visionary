"""
Classical Computer Vision: Image Preprocessing & Illumination Normalization.

Includes:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in CIE LAB space
- Bilateral filtering (edge-preserving denoising)
- Gaussian smoothing
"""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    For color (RGB) images, equalization is applied strictly to the Lightness (L)
    channel in CIE LAB color space to preserve chromaticity and prevent color distortion.
    
    Args:
        image: Input image (RGB or grayscale uint8 array).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization (rows, cols).
        
    Returns:
        Illumination-normalized image of identical shape and dtype.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)
    elif image.ndim == 3 and image.shape[2] == 3:
        # Convert RGB to CIE LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE strictly to the Lightness channel
        l_equalized = clahe.apply(l_channel)
        
        # Merge back and convert back to RGB
        lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        # Handle RGBA images
        rgb = image[:, :, :3]
        alpha = image[:, :, 3:]
        equalized_rgb = apply_clahe(rgb, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return np.concatenate([equalized_rgb, alpha], axis=2)
    else:
        raise ValueError(f"Unsupported image shape for CLAHE: {image.shape}")


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Applies edge-preserving bilateral filtering to reduce noise while maintaining sharp edges.
    
    Args:
        image: Input image (RGB or grayscale uint8 array).
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
        
    Returns:
        Denoised image with sharp structural boundaries preserved.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def gaussian_blur(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma_x: float = 1.0,
) -> np.ndarray:
    """
    Applies Gaussian blurring to suppress high-frequency noise prior to gradient extraction.
    
    Args:
        image: Input image.
        ksize: Gaussian kernel size (width, height) - must be odd numbers.
        sigma_x: Gaussian kernel standard deviation in X direction.
        
    Returns:
        Gaussian blurred image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigma_x)
