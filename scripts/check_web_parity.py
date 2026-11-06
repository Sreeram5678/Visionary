"""
Compare the in-browser pipeline (web/pipeline.js, run in Node by
web/test/parity.mjs) against the Python engine on the same test images.

Python reference config = what the browser runs: YOLO ROI + adaptive CLAHE
gate + flip TTA, INT8 classifier, no classical head (ablation row G, INT8).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.metrics import compute_iou  # noqa: E402

web = json.loads((ROOT / "data/web_parity.json").read_text())
te = pd.read_csv(ROOT / "data/prepared/test.csv")
V = np.load(ROOT / "data/features/views_convnext_tiny_224_int8_test.npz")
dets = json.loads((ROOT / "data/features/yolo_test.json").read_text())

rows = np.array([w["row"] for w in web])
y = te.label.values[rows]
g = V["gate"][rows]
crop = np.where(g[:, None], V["crop_c"][rows], V["crop"][rows])
flip = np.where(g[:, None], V["crop_flip_c"][rows], V["crop_flip"][rows])
py_p = 0.5 * (crop + flip)
py_pred = py_p.argmax(1)
js_pred = np.array([w["pred"] for w in web])
js_top5 = np.array([w["top5"] for w in web])

ious = [compute_iou(w["box"], dets[r]["box"]) if (w["box"] and dets[r]["box"]) else float(w["box"] == dets[r]["box"])
        for w, r in zip(web, rows)]
res = {
    "n": int(len(web)),
    "top1_agreement_js_vs_python": round(float((js_pred == py_pred).mean() * 100), 2),
    "js_top1": round(float((js_pred == y).mean() * 100), 2),
    "python_top1_same_images": round(float((py_pred == y).mean() * 100), 2),
    "js_top5": round(float((js_top5 == y[:, None]).any(1).mean() * 100), 2),
    "gate_agreement": round(float(np.mean([w["gate"] == bool(gg) for w, gg in zip(web, g)]) * 100), 2),
    "box_iou_js_vs_python_mean": round(float(np.mean(ious)), 4),
    "js_mean_total_ms_node_1thread": round(float(np.mean([w["ms"]["total_ms"] for w in web])), 1),
}
(ROOT / "weights/web_parity.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
