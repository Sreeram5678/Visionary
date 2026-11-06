"""
Hybrid inference engine: Classical CV + YOLO detection + fine-grained DL.

Pipeline for one uploaded image
  1. Validate / decode, cap resolution.
  2. YOLOv8n localisation (COCO car/truck/bus) -> primary vehicle box.
  3. Adaptive illumination gate: if the ROI is dark or low-contrast,
     CLAHE (LAB L-channel) is applied before classification.
  4. Deep classifier on (a) the ROI crop, (b) its horizontal flip, and
     (c) the full frame; probabilities fused with weights tuned on val.
  5. Optional late fusion with the classical HOG/LBP/HSV classifier.
  6. Temperature-calibrated confidence + low-confidence handling.

All weights / thresholds come from weights/models.json so the evaluation
script and the live app run exactly the same code path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.classical.preprocessing import apply_clahe
from src.classification.class_names import get_metadata
from src.detection.cropper import crop_roi

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VEHICLE_COCO = {2: "car", 5: "bus", 7: "truck"}
MAX_SIDE = 1600


def softmax(z: np.ndarray, axis=-1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def loglinear_fusion(p_deep: np.ndarray, p_cls: np.ndarray, w: float, t: float = 1.0) -> np.ndarray:
    """Product-of-experts fusion: log p ∝ (log p_deep + w·log p_classical) / t.

    Unlike a weighted average, a weak expert can only *re-rank* among classes the
    deep model already considers plausible; it cannot flatten the distribution,
    so Top-5 and calibration are preserved.
    """
    z = (np.log(p_deep + 1e-9) + w * np.log(p_cls + 1e-9)) / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def illumination_stats(rgb: np.ndarray) -> Dict[str, float]:
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    return {"mean_luma": float(L.mean()), "contrast": float(L.std())}


def needs_clahe(stats: Dict[str, float], dark: float = 70.0, flat: float = 35.0) -> bool:
    return stats["mean_luma"] < dark or stats["contrast"] < flat


# --------------------------------------------------------------------------- #
# Classifier backends (PyTorch fp32 / ONNX INT8) behind one interface
# --------------------------------------------------------------------------- #
class TorchBackend:
    def __init__(self, ckpt_path: Path, device: Optional[str] = None):
        import torch
        from src.classification.model import VehicleClassifier

        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.img_size = int(ck.get("img_size", 224))
        self.temperature = float(ck.get("temperature", 1.0))
        self.model = VehicleClassifier(ck["backbone"], num_classes=196, pretrained=False)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
        self.model.load_state_dict(sd, strict=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.size_mb = ckpt_path.stat().st_size / 2**20
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

    def logits(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.torch.from_numpy(batch).to(self.device)
            if self.device.type == "cuda":
                with self.torch.autocast("cuda", dtype=self.torch.float16):
                    return self.model(x).float().cpu().numpy()
            return self.model(x).numpy()


class OnnxBackend:
    def __init__(self, onnx_path: Path, meta: dict):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.img_size = int(meta.get("img_size", 224))
        self.temperature = float(meta.get("temperature", 1.0))
        self.size_mb = onnx_path.stat().st_size / 2**20
        self.params_m = float(meta.get("params_m", 0))

    def logits(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.inp: batch.astype(np.float32)})[0]


# --------------------------------------------------------------------------- #
@dataclass
class HybridConfig:
    w_full: float = 0.25          # weight of full-frame view vs ROI crop
    w_classical: float = 0.0      # weight of classical branch in log-linear fusion
    fusion_t: float = 1.0         # temperature re-fitted on val after fusion
    use_flip_tta: bool = True
    use_clahe_gate: bool = True
    use_detector: bool = True
    low_conf: float = 0.50        # below -> "uncertain" handling
    det_conf: float = 0.25


@dataclass
class Detection:
    box: Optional[List[float]] = None
    conf: float = 0.0
    label: str = ""
    all_boxes: List[dict] = field(default_factory=list)
    ms: float = 0.0


class HybridEngine:
    def __init__(self, backend, cfg: HybridConfig, classical=None, yolo_weights: str = "yolov8n.pt"):
        self.backend = backend
        self.cfg = cfg
        self.classical = classical
        self._yolo = None
        self.yolo_weights = yolo_weights

    # ---- detection ------------------------------------------------------- #
    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            p = WEIGHTS / self.yolo_weights
            self._yolo = YOLO(str(p) if p.exists() else self.yolo_weights)
        return self._yolo

    def detect(self, rgb: np.ndarray) -> Detection:
        t = time.perf_counter()
        res = self.yolo.predict(rgb[:, :, ::-1], conf=self.cfg.det_conf, verbose=False,
                                classes=list(VEHICLE_COCO), imgsz=640)[0]
        det = Detection(ms=(time.perf_counter() - t) * 1000)
        if res.boxes is None or len(res.boxes) == 0:
            return det
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        # Primary vehicle = largest box, weighted by confidence.
        i = int(np.argmax(area * conf))
        det.box = [float(v) for v in xyxy[i]]
        det.conf = float(conf[i])
        det.label = VEHICLE_COCO.get(int(cls[i]), "vehicle")
        det.all_boxes = [{"box": [round(float(v), 1) for v in b], "conf": round(float(c), 3),
                          "label": VEHICLE_COCO.get(int(k), "vehicle")}
                         for b, c, k in zip(xyxy, conf, cls)]
        return det

    # ---- classification -------------------------------------------------- #
    def _prep(self, rgb: np.ndarray) -> np.ndarray:
        s = self.backend.img_size
        x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return x.transpose(2, 0, 1)

    def classify_views(self, roi: np.ndarray, full: Optional[np.ndarray]) -> np.ndarray:
        views = [self._prep(roi)]
        if self.cfg.use_flip_tta:
            views.append(self._prep(roi[:, ::-1]))
        n_roi = len(views)
        if full is not None and self.cfg.w_full > 0:
            views.append(self._prep(full))
        logits = self.backend.logits(np.stack(views)) / self.backend.temperature
        p = softmax(logits)
        p_roi = p[:n_roi].mean(0)
        if len(views) > n_roi:
            return (1 - self.cfg.w_full) * p_roi + self.cfg.w_full * p[n_roi]
        return p_roi

    @staticmethod
    def _top_k(probs: np.ndarray, k: int) -> List[dict]:
        out = []
        for i in np.argsort(probs)[::-1][:k]:
            m = get_metadata(int(i))
            out.append({"class_id": int(i), "name": m.full_name, "make": m.make, "model": m.model,
                        "year": m.year, "body_style": m.body_style,
                        "confidence": round(float(probs[i]) * 100, 2)})
        return out

    @staticmethod
    def select_vehicles(det: "Detection", W: int, H: int, min_side: int = 48,
                        min_area_frac: float = 0.01, max_n: int = 6) -> List[dict]:
        """Primary first, then other vehicles by area*conf; drops duplicates
        (IoU > 0.5 or > 80 % inside a kept box) and boxes too small to name.
        Mirrors Pipeline.selectVehicles in web/pipeline.js."""
        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        def inter(a, b):
            return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

        ranked = sorted(det.all_boxes, key=lambda b: area(b["box"]) * b["conf"], reverse=True)
        keep = [{"box": det.box, "conf": round(det.conf, 3), "label": det.label}] if det.box else []
        for b in ranked:
            w, h = b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]
            if min(w, h) < min_side or w * h < min_area_frac * W * H:
                continue
            if any(inter(k["box"], b["box"]) / max(1e-6, area(k["box"]) + area(b["box"]) - inter(k["box"], b["box"])) > 0.5
                   or inter(k["box"], b["box"]) / max(1e-6, area(b["box"])) > 0.8 for k in keep):
                continue
            keep.append(b)
            if len(keep) >= max_n:
                break
        return keep

    # ---- full analysis --------------------------------------------------- #
    def analyze(self, rgb: np.ndarray, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

        timings = {}
        det = self.detect(rgb) if self.cfg.use_detector else Detection()
        timings["detection_ms"] = det.ms
        roi = crop_roi(rgb, det.box, padding_ratio=0.05) if det.box else rgb

        t = time.perf_counter()
        illum = illumination_stats(roi)
        clahe_applied = self.cfg.use_clahe_gate and needs_clahe(illum)
        roi_in = apply_clahe(roi, clip_limit=2.5) if clahe_applied else roi
        full_in = None
        if det.box is not None:
            full_in = apply_clahe(rgb, clip_limit=2.5) if clahe_applied else rgb
        timings["classical_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        probs = self.classify_views(roi_in, full_in)
        timings["classifier_ms"] = (time.perf_counter() - t) * 1000

        if self.classical is not None and self.cfg.w_classical > 0 and self.classical.available:
            t = time.perf_counter()
            pc = self.classical.predict_proba(roi)
            probs = loglinear_fusion(probs, pc, self.cfg.w_classical, self.cfg.fusion_t)
            timings["classical_head_ms"] = (time.perf_counter() - t) * 1000

        preds = self._top_k(probs, top_k)
        top = [p["class_id"] for p in preds]
        conf = float(probs[top[0]])
        margin = float(probs[top[0]] - probs[top[1]])

        if det.box is None and conf < self.cfg.low_conf:
            status = "no_vehicle"
            message = "No vehicle detected in this image. Try a clearer photo of a car."
        elif conf < self.cfg.low_conf:
            status = "low_confidence"
            message = (f"Low confidence ({conf*100:.0f}%). The car may be a model outside the 196 "
                       f"Stanford Cars classes, heavily occluded, or shot from an unusual angle — "
                       f"see the top-5 alternatives.")
        else:
            status = "ok"
            message = "Vehicle identified." if det.box else \
                "Detector found no vehicle box; classified the full frame."

        # ---- every other vehicle in the scene (single view each, batched) ----
        vehicles = [{"index": 1, "box": det.box, "det_conf": round(det.conf, 3), "label": det.label,
                     "top5": preds, "prediction": preds[0], "confidence": preds[0]["confidence"],
                     "status": status, "clahe_applied": clahe_applied}]
        others = self.select_vehicles(det, rgb.shape[1], rgb.shape[0])[1:] if det.box else []
        if others:
            t = time.perf_counter()
            rois = []
            for v in others:
                r = crop_roi(rgb, v["box"], padding_ratio=0.05)
                g = self.cfg.use_clahe_gate and needs_clahe(illumination_stats(r))
                rois.append((apply_clahe(r, clip_limit=2.5) if g else r, g))
            logits = self.backend.logits(np.stack([self._prep(r) for r, _ in rois])) / self.backend.temperature
            for j, (v, (_, g), p) in enumerate(zip(others, rois, softmax(logits))):
                tk = self._top_k(p, top_k)
                vehicles.append({"index": j + 2, "box": v["box"], "det_conf": v["conf"], "label": v["label"],
                                 "top5": tk, "prediction": tk[0], "confidence": tk[0]["confidence"],
                                 "status": "ok" if p.max() >= self.cfg.low_conf else "low_confidence",
                                 "clahe_applied": bool(g)})
            timings["other_vehicles_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {
            "vehicles": vehicles,
            "status": status, "message": message,
            "prediction": preds[0], "top5": preds,
            "confidence": round(conf * 100, 2), "margin": round(margin * 100, 2),
            "detection": {"box": det.box, "conf": round(det.conf, 3), "label": det.label,
                          "all": det.all_boxes, "image_size": [rgb.shape[1], rgb.shape[0]]},
            "illumination": {**{k: round(v, 1) for k, v in illum.items()}, "clahe_applied": clahe_applied},
            "timings": {k: round(v, 1) for k, v in timings.items()},
            "_roi": roi, "_roi_in": roi_in, "_rgb": rgb,
        }


# --------------------------------------------------------------------------- #
def load_registry() -> dict:
    p = WEIGHTS / "models.json"
    return json.loads(p.read_text()) if p.exists() else {"default": None, "models": {}}


WEB_MODELS = ROOT / "web" / "models"


def resolve_model_file(spec: dict) -> Optional[Path]:
    """weights/<file>, else the INT8 copy shipped with the web app (web/models/)."""
    for p in (WEIGHTS / spec["file"], WEB_MODELS / spec["file"]):
        if p.exists():
            return p
    return None


def available_registry(reg: Optional[dict] = None) -> dict:
    """Drop variants whose weight file is missing (e.g. the >100 MB FP32 ONNX files
    are not in git); keep the default if possible, else the first available."""
    reg = dict(reg or load_registry())
    models = {k: m for k, m in reg["models"].items() if resolve_model_file(m)}
    if not models:
        return {**reg, "models": {}}
    default = reg.get("default") if reg.get("default") in models else next(iter(models))
    # Prefer the INT8 sibling of a missing full-precision default.
    if reg.get("default") not in models and f"{reg.get('default')}_int8" in models:
        default = f"{reg.get('default')}_int8"
    return {**reg, "default": default, "models": models}


def build_engine(key: str, registry: Optional[dict] = None, device: Optional[str] = None) -> HybridEngine:
    from src.hybrid.classical_head import ClassicalHead

    reg = registry or load_registry()
    spec = reg["models"][key]
    path = resolve_model_file(spec)
    if path is None:
        raise FileNotFoundError(f"weights for {key} not found in weights/ or web/models/")
    if spec["runtime"] == "onnx":
        backend = OnnxBackend(path, spec)
    else:
        backend = TorchBackend(path, device=device)
    cfg = HybridConfig(**spec.get("hybrid", reg.get("hybrid", {})))
    classical = ClassicalHead(str(WEIGHTS / "classical_head.joblib"))
    return HybridEngine(backend, cfg, classical=classical)
