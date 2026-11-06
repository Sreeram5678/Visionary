"""End-to-end API tests: happy path + every bad-input path the judges may try."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "weights" / "models.json"
pytestmark = pytest.mark.skipif(not REG.exists(), reason="trained weights not present")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def png(w, h, color=(30, 120, 200)):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, "PNG")
    return b.getvalue()


def test_index_and_models(client):
    assert client.get("/").status_code == 200
    m = client.get("/api/models").json()
    assert m["default"] in {x["key"] for x in m["models"]}


def test_empty_file(client):
    r = client.post("/api/predict", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400 and r.json()["status"] == "error"


def test_not_an_image(client):
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_tiny_image(client):
    r = client.post("/api/predict", files={"file": ("a.png", png(8, 8), "image/png")})
    assert r.status_code == 422


def test_no_vehicle(client):
    r = client.post("/api/predict", files={"file": ("sky.png", png(400, 300), "image/png")})
    assert r.status_code == 200 and r.json()["status"] == "no_vehicle"


def test_car_prediction(client):
    test_csv = ROOT / "data/prepared/test.csv"
    if not test_csv.exists():
        pytest.skip("dataset not prepared")
    import pandas as pd
    row = pd.read_csv(test_csv).iloc[0]
    data = (ROOT / "data/prepared" / row.path).read_bytes()
    r = client.post("/api/predict", files={"file": ("car.jpg", data, "image/jpeg")}).json()
    assert r["status"] in {"ok", "low_confidence"}
    assert len(r["top5"]) == 5 and r["detection"]["box"] is not None
    assert 0 <= r["confidence"] <= 100
    assert {"make", "model", "year"} <= set(r["prediction"])


def test_web_app_served_with_isolation_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Visionary" in r.text
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cross-origin-embedder-policy"] == "require-corp"


def test_multiple_vehicles(client):
    img = ROOT / "data/prepared/images/test/test_00631.jpg"  # dealership lot, several cars
    if not img.exists():
        pytest.skip("dataset not prepared")
    r = client.post("/api/predict", files={"file": ("lot.jpg", img.read_bytes(), "image/jpeg")}).json()
    v = r["vehicles"]
    assert len(v) >= 3
    assert [x["index"] for x in v] == list(range(1, len(v) + 1))
    assert v[0]["prediction"]["name"] == "Dodge Charger Sedan 2012"
