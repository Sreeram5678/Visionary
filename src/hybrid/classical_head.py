"""
Classical-only fine-grained classifier (baseline + fusion branch).

Hand-crafted descriptor on the vehicle ROI:
  * HOG  (shape / silhouette / grille + headlight gradients)  — 1764 dims
  * LBP  (local texture, uniform, P=24, R=3)                   —   26 dims
  * HSV colour histogram (8x8x8)                               —  512 dims
Standardised -> PCA(256) -> multinomial Logistic Regression (probabilistic,
so its output can be fused with the deep model's softmax).
"""

from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from src.classical.features import extract_color_histogram, extract_lbp_texture
from src.classical.preprocessing import apply_clahe

HOG_SIZE = (128, 128)


def classical_descriptor(roi_rgb: np.ndarray) -> np.ndarray:
    roi = apply_clahe(roi_rgb)
    gray = cv2.cvtColor(cv2.resize(roi, HOG_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    small = cv2.resize(roi, (160, 160), interpolation=cv2.INTER_AREA)
    lbp = extract_lbp_texture(small).astype(np.float32)
    col = extract_color_histogram(small).astype(np.float32)
    return np.concatenate([h, lbp, col])


class ClassicalHead:
    def __init__(self, path: Optional[str] = None):
        self.pipe = joblib.load(path) if path and Path(path).exists() else None

    @property
    def available(self) -> bool:
        return self.pipe is not None

    def predict_proba(self, roi_rgb: np.ndarray) -> np.ndarray:
        feat = classical_descriptor(roi_rgb)[None]
        return self.pipe.predict_proba(feat)[0]
