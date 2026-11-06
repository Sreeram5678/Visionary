"""
Unit tests for Fine-Grained Vehicle Classification Subsystem.
Tests Stanford Cars 196 class taxonomy, label decomposition,
ConvNeXt model forward pass, penultimate embedding extraction,
and VehicleClassifierEngine calibrated prediction.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.classification import (
    CAR_METADATA_REGISTRY,
    STANFORD_CARS_CLASSES,
    CarMetadata,
    CarPrediction,
    ClassificationResult,
    VehicleClassifier,
    VehicleClassifierEngine,
    get_all_makes,
    get_class_name,
    get_classes_by_make,
    get_metadata,
    parse_car_name,
)


# ---------------------------------------------------------------------------
# Taxonomy & Parser Tests
# ---------------------------------------------------------------------------

def test_stanford_cars_class_count():
    assert len(STANFORD_CARS_CLASSES) == 196
    # Verify no duplicates
    assert len(set(STANFORD_CARS_CLASSES)) == 196


def test_stanford_cars_class_lookup():
    assert get_class_name(0) == "AM General Hummer SUV 2000"
    assert get_class_name(14) == "Audi R8 Coupe 2012"
    with pytest.raises(IndexError):
        get_class_name(196)
    with pytest.raises(IndexError):
        get_class_name(-1)


def test_parse_car_name_multiword_makes():
    # Multi-word make test: AM General
    meta_hummer = parse_car_name("AM General Hummer SUV 2000", class_id=0)
    assert meta_hummer.make == "AM General"
    assert meta_hummer.year == 2000
    assert meta_hummer.body_style == "SUV"

    # Multi-word make test: Aston Martin
    meta_aston = parse_car_name("Aston Martin V8 Vantage Coupe 2012", class_id=8)
    assert meta_aston.make == "Aston Martin"
    assert meta_aston.year == 2012
    assert meta_aston.body_style == "Coupe"

    # Multi-word make test: Mercedes-Benz
    meta_merc = parse_car_name("Mercedes-Benz C-Class Sedan 2012")
    assert meta_merc.make == "Mercedes-Benz"
    assert meta_merc.year == 2012
    assert meta_merc.body_style == "Sedan"


def test_parse_car_name_single_word_makes():
    meta_audi = parse_car_name("Audi R8 Coupe 2012", class_id=14)
    assert meta_audi.make == "Audi"
    assert meta_audi.year == 2012
    assert meta_audi.body_style == "Coupe"
    assert "R8" in meta_audi.model


def test_get_all_makes():
    makes = get_all_makes()
    assert len(makes) >= 40
    assert "Audi" in makes
    assert "BMW" in makes
    assert "Chevrolet" in makes
    assert "Ferrari" in makes


def test_get_classes_by_make():
    audi_cars = get_classes_by_make("Audi")
    assert len(audi_cars) > 0
    for car in audi_cars:
        assert car.make == "Audi"


# ---------------------------------------------------------------------------
# Deep Learning Model Tests
# ---------------------------------------------------------------------------

def test_vehicle_classifier_model_shapes():
    # Instantiate with pretrained=False for fast local testing
    model = VehicleClassifier(backbone_name="convnext_tiny", num_classes=196, pretrained=False)
    assert model.embedding_dim == 768

    dummy_input = torch.randn(2, 3, 224, 224)

    # Standard forward pass
    logits = model(dummy_input, return_embedding=False)
    assert logits.shape == (2, 196)

    # Forward pass requesting penultimate embedding
    logits, embedding = model(dummy_input, return_embedding=True)
    assert logits.shape == (2, 196)
    assert embedding.shape == (2, 768)


# ---------------------------------------------------------------------------
# Inference Engine Tests
# ---------------------------------------------------------------------------

def test_classifier_engine_predict_numpy():
    engine = VehicleClassifierEngine(backbone_name="convnext_tiny", device="cpu", temperature=1.5)
    
    # 224x224 RGB test image
    synthetic_car = np.full((224, 224, 3), 120, dtype=np.uint8)
    # Simulated vehicle body
    synthetic_car[60:160, 30:190] = [200, 40, 40]

    result = engine.predict(synthetic_car, top_k=5)

    assert isinstance(result, ClassificationResult)
    assert isinstance(result.top1, CarPrediction)
    assert len(result.top5) == 5
    assert result.embedding.shape == (768,)
    assert result.inference_time_ms > 0

    # Top-1 should have highest probability
    assert result.top1.probability >= result.top5[1].probability
    # Probabilities in range [0, 1]
    for p in result.top5:
        assert 0.0 <= p.probability <= 1.0

    # Test dictionary serialization
    d = result.to_dict()
    assert "top1" in d
    assert "top5" in d
    assert d["embedding_dim"] == 768


def test_classifier_engine_pil_input():
    engine = VehicleClassifierEngine(backbone_name="convnext_tiny", device="cpu")
    pil_img = Image.new("RGB", (250, 200), color="blue")
    result = engine.predict(pil_img, top_k=3)
    assert len(result.top5) == 3
    assert result.top1.class_id in range(196)
