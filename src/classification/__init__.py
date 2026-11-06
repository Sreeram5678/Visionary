"""
Fine-Grained Vehicle Classification Subsystem for Visionary.

Provides Stanford Cars 196 class taxonomy, deep transfer learning model,
calibrated probability prediction, and penultimate feature embeddings.
"""

from src.classification.class_names import (
    CAR_METADATA_REGISTRY,
    STANFORD_CARS_CLASSES,
    CarMetadata,
    get_all_makes,
    get_class_name,
    get_classes_by_make,
    get_metadata,
    parse_car_name,
)
from src.classification.classifier import (
    CarPrediction,
    ClassificationResult,
    VehicleClassifierEngine,
)
from src.classification.model import VehicleClassifier

__all__ = [
    "STANFORD_CARS_CLASSES",
    "CAR_METADATA_REGISTRY",
    "CarMetadata",
    "get_class_name",
    "get_metadata",
    "get_all_makes",
    "get_classes_by_make",
    "parse_car_name",
    "VehicleClassifier",
    "VehicleClassifierEngine",
    "CarPrediction",
    "ClassificationResult",
]
