"""
Unit tests for the Classical Computer Vision Subsystem.
Tests preprocessing (CLAHE, bilateral, Gaussian), edge/contour extraction,
color/texture descriptors, and end-to-end ClassicalPipeline.
"""

import numpy as np
import pytest
from PIL import Image

from src.classical import (
    ClassicalPipeline,
    ClassicalResult,
    apply_clahe,
    denoise_bilateral,
    detect_edges_canny,
    extract_classical_feature_vector,
    extract_color_histogram,
    extract_contours,
    extract_geometric_features,
    extract_lbp_texture,
    gaussian_blur,
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Creates a synthetic 200x300 RGB test image with shapes and gradient."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Background gradient
    for y in range(200):
        img[y, :, 0] = int((y / 200.0) * 128)
        img[y, :, 1] = 64
        img[y, :, 2] = int(((200 - y) / 200.0) * 128)

    # Add a bright simulated vehicle body rectangle (x: 50..250, y: 70..150) -> w: 200, h: 80
    img[70:150, 50:250] = [220, 30, 30]

    # Add two dark simulated wheels
    img[140:170, 70:110] = [20, 20, 20]
    img[140:170, 190:230] = [20, 20, 20]

    return img


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Creates a 150x150 grayscale test image."""
    img = np.full((150, 150), 100, dtype=np.uint8)
    img[40:110, 40:110] = 230  # bright central square
    return img


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_apply_clahe_rgb(sample_rgb_image):
    result = apply_clahe(sample_rgb_image, clip_limit=2.0)
    assert result.shape == sample_rgb_image.shape
    assert result.dtype == np.uint8
    # Contrast should be modified
    assert not np.array_equal(result, sample_rgb_image)


def test_apply_clahe_grayscale(sample_gray_image):
    result = apply_clahe(sample_gray_image)
    assert result.shape == sample_gray_image.shape
    assert result.dtype == np.uint8


def test_apply_clahe_invalid():
    with pytest.raises(ValueError):
        apply_clahe(None)
    with pytest.raises(ValueError):
        apply_clahe(np.array([]))


def test_denoise_bilateral(sample_rgb_image):
    # Add Gaussian noise to test image
    noise = np.random.normal(0, 15, sample_rgb_image.shape).astype(np.int16)
    noisy = np.clip(sample_rgb_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    denoised = denoise_bilateral(noisy, d=5, sigma_color=50, sigma_space=50)
    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.uint8
    
    # Flat background region should have lower standard deviation after bilateral filter
    noisy_std = np.std(noisy[10:40, 10:40])
    denoised_std = np.std(denoised[10:40, 10:40])
    assert denoised_std < noisy_std


def test_gaussian_blur(sample_rgb_image):
    blurred = gaussian_blur(sample_rgb_image, ksize=(5, 5))
    assert blurred.shape == sample_rgb_image.shape
    assert blurred.dtype == np.uint8


# ---------------------------------------------------------------------------
# Edge & Contour Tests
# ---------------------------------------------------------------------------

def test_detect_edges_canny(sample_rgb_image):
    edges = detect_edges_canny(sample_rgb_image)
    assert edges.shape == sample_rgb_image.shape[:2]
    assert edges.dtype == np.uint8
    # Binary output: only 0 and 255
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})
    # Must have detected edges for our rectangle
    assert edges.sum() > 0


def test_detect_edges_canny_solid_image():
    # Completely black or solid image should not crash
    solid = np.zeros((100, 100, 3), dtype=np.uint8)
    edges = detect_edges_canny(solid)
    assert edges.shape == (100, 100)
    assert np.all(edges == 0)


def test_extract_contours(sample_rgb_image):
    edges = detect_edges_canny(sample_rgb_image)
    contours, vis = extract_contours(edges, min_area=50.0)
    
    assert isinstance(contours, list)
    assert len(contours) > 0
    assert vis.shape == (sample_rgb_image.shape[0], sample_rgb_image.shape[1], 3)
    assert vis.dtype == np.uint8


def test_extract_geometric_features():
    # Synthetic clean rectangle: width 100, height 50 -> aspect ratio = 2.0
    img_shape = (200, 300)
    rect_contour = np.array([
        [[50, 50]],
        [[150, 50]],
        [[150, 100]],
        [[50, 100]],
    ], dtype=np.int32)

    metrics = extract_geometric_features([rect_contour], img_shape)
    
    assert "aspect_ratio" in metrics
    assert "compactness" in metrics
    assert "extent" in metrics
    assert "solidity" in metrics
    assert pytest.approx(metrics["aspect_ratio"], rel=0.05) == 2.0
    assert pytest.approx(metrics["extent"], rel=0.05) == 1.0
    assert pytest.approx(metrics["solidity"], rel=0.05) == 1.0


def test_extract_geometric_features_empty():
    metrics = extract_geometric_features([], (200, 200))
    assert metrics["aspect_ratio"] == 1.0
    assert metrics["compactness"] == 0.0


# ---------------------------------------------------------------------------
# Feature Descriptors Tests
# ---------------------------------------------------------------------------

def test_extract_color_histogram(sample_rgb_image):
    hist = extract_color_histogram(sample_rgb_image, bins=(8, 8, 8))
    # 8 * 8 * 8 = 512 dimensions
    assert hist.shape == (512,)
    assert hist.dtype == np.float32
    # L2-normalized: norm should be close to 1.0
    assert pytest.approx(np.linalg.norm(hist), abs=1e-5) == 1.0


def test_extract_color_histogram_distinct():
    # Pure red vs pure blue images should produce orthogonal/distinct histograms
    red = np.zeros((100, 100, 3), dtype=np.uint8)
    red[:, :] = [255, 0, 0]
    
    blue = np.zeros((100, 100, 3), dtype=np.uint8)
    blue[:, :] = [0, 0, 255]

    h_red = extract_color_histogram(red)
    h_blue = extract_color_histogram(blue)

    # Dot product between pure red and pure blue should be ~ 0
    cosine_sim = np.dot(h_red, h_blue)
    assert cosine_sim < 0.1


def test_extract_lbp_texture(sample_rgb_image):
    lbp_hist = extract_lbp_texture(sample_rgb_image, num_points=24, radius=3, method="uniform")
    # For uniform LBP with P=24, number of bins = P + 2 = 26
    assert lbp_hist.shape == (26,)
    assert lbp_hist.dtype == np.float32
    # L1-normalized probability distribution: sum should be close to 1.0
    assert pytest.approx(float(lbp_hist.sum()), abs=1e-5) == 1.0


def test_extract_classical_feature_vector(sample_rgb_image):
    feat, metrics = extract_classical_feature_vector(sample_rgb_image)
    # Total dim: 512 (color) + 26 (texture) + 6 (geometry) = 544
    assert feat.shape == (544,)
    assert feat.dtype == np.float32
    assert isinstance(metrics, dict)
    assert "aspect_ratio" in metrics


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

def test_classical_pipeline_end_to_end(sample_rgb_image):
    pipeline = ClassicalPipeline()
    result = pipeline.process(sample_rgb_image)

    assert isinstance(result, ClassicalResult)
    assert result.features.shape == (544,)
    assert isinstance(result.metrics, dict)
    
    # Check all expected visualizations exist
    for key in ["raw", "clahe", "denoised", "edges", "contours"]:
        assert key in result.visualizations
        vis_img = result.visualizations[key]
        assert vis_img.shape[:2] == sample_rgb_image.shape[:2]
        assert vis_img.dtype == np.uint8


def test_classical_pipeline_pil_input(sample_rgb_image):
    pipeline = ClassicalPipeline()
    pil_img = Image.fromarray(sample_rgb_image)
    result = pipeline.process(pil_img)
    assert result.features.shape == (544,)


def test_classical_pipeline_dict_serialization(sample_rgb_image):
    pipeline = ClassicalPipeline()
    result = pipeline.process(sample_rgb_image)
    d = result.to_dict()
    assert d["features_dim"] == 544
    assert "metrics" in d
    assert "available_visualizations" in d
