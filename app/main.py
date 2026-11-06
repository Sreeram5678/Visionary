"""
Visionary — FastAPI service.

GET  /                 -> the in-browser app (web/), same build as the live HF Space
                          (served with COOP/COEP so WebAssembly can use threads)
GET  /api/models       -> available classifier variants + default
GET  /api/health       -> liveness
GET  /api/benchmarks   -> evaluation results (weights/eval_results.json)
POST /api/predict      -> multipart image upload (+ optional `model` field)
"""

import base64
import io
import json
import logging
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical.edges import detect_edges_canny, extract_contours  # noqa: E402
from src.classical.preprocessing import apply_clahe, denoise_bilateral  # noqa: E402
from src.hybrid.engine import WEIGHTS, available_registry, build_engine  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visionary")

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {"JPEG", "PNG", "WEBP", "BMP", "MPO"}

app = FastAPI(title="Visionary — Fine-Grained Vehicle Recognition")
WEB = ROOT / "web"


@app.middleware("http")
async def isolation_headers(request, call_next):
    """Cross-origin isolation lets the browser app use multi-threaded WASM."""
    resp = await call_next(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return resp

_registry = available_registry()
_engines = {}
_lock = threading.Lock()


def get_engine(key: str):
    with _lock:
        if key not in _engines:
            log.info("loading model %s", key)
            _engines[key] = build_engine(key, _registry, device=os.environ.get("VISIONARY_DEVICE", "cpu"))
        return _engines[key]


@app.on_event("startup")
def warmup():
    key = _registry.get("default")
    if key:
        eng = get_engine(key)
        eng.analyze(np.full((240, 320, 3), 128, np.uint8))  # warm YOLO + classifier
        log.info("warm-up done for %s", key)


def to_b64(rgb: np.ndarray, max_side: int = 512, quality: int = 82) -> str:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def stages(roi: np.ndarray) -> list:
    """Intermediate classical-CV visualisations shown in the UI."""
    clahe = apply_clahe(roi, clip_limit=2.5)
    den = denoise_bilateral(clahe)
    edges = detect_edges_canny(den)
    _, contours = extract_contours(edges, min_area=50.0)
    return [
        {"name": "Vehicle ROI", "desc": "YOLOv8 crop of the primary vehicle", "img": to_b64(roi, 384)},
        {"name": "CLAHE", "desc": "Contrast-limited equalisation on LAB lightness", "img": to_b64(clahe, 384)},
        {"name": "Bilateral denoise", "desc": "Edge-preserving smoothing", "img": to_b64(den, 384)},
        {"name": "Canny edges", "desc": "Auto-threshold edges (median ± 33%)", "img": to_b64(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 384)},
        {"name": "Contours", "desc": "Structural contours used for geometry features", "img": to_b64(contours, 384)},
    ]


def err(status: int, msg: str):
    return JSONResponse(status_code=status, content={"status": "error", "message": msg})


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": list(_engines)}


@app.get("/api/models")
def models():
    out = []
    for k, m in _registry["models"].items():
        out.append({"key": k, "name": m["name"], "desc": m.get("desc", ""),
                    "top1": m.get("top1"), "latency_ms": m.get("latency_ms")})
    return {"default": _registry.get("default"), "models": out}


@app.get("/api/benchmarks")
def benchmarks():
    p = WEIGHTS / "eval_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form(None)):
    data = await file.read()
    if not data:
        return err(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        return err(413, "Image is larger than 15 MB. Please upload a smaller image.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return err(415, "That file isn't a readable image. Please upload a JPG, PNG or WEBP.")
    if fmt not in ALLOWED:
        return err(415, f"Unsupported image format ({fmt}). Use JPG, PNG or WEBP.")
    if min(img.size) < 32:
        return err(422, "Image is too small (minimum 32×32 pixels).")

    key = model if model in _registry["models"] else _registry["default"]
    try:
        engine = get_engine(key)
        rgb = np.asarray(img)
        res = engine.analyze(rgb)
    except Exception as e:  # never leak a stack trace to the UI
        log.exception("inference failed")
        return err(500, f"Inference failed: {type(e).__name__}. Please try another image.")

    rgb = res.pop("_rgb"); roi = res.pop("_roi"); res.pop("_roi_in")
    res["model"] = {"key": key, "name": _registry["models"][key]["name"]}
    res["image"] = to_b64(rgb, 1024, 85)
    res["stages"] = stages(roi) if res["status"] != "no_vehicle" else []
    return res


# Mounted last so /api/* routes take precedence.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
