"""INT8 vs FP32 accuracy of the browser configuration on a stratified test subset.

Browser config = YOLO ROI + adaptive CLAHE gate + flip TTA (no classical head).
Usage: python scripts/int8_subset_check.py --name convnext_small_320 --n 2000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classical.preprocessing import apply_clahe  # noqa: E402
from src.hybrid.engine import (WEIGHTS, HybridConfig, HybridEngine, OnnxBackend,  # noqa: E402
                               illumination_stats, needs_clahe, softmax)
from scripts.evaluate import crop  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="convnext_small_320"); ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
te = pd.read_csv(ROOT / "data/prepared/test.csv")
sub = te.groupby("label").sample(n=max(1, a.n // 196), random_state=1)
idx = sub.index.values
dets = json.loads((ROOT / "data/features/yolo_test.json").read_text())
meta = json.loads((WEIGHTS / f"{a.name}_int8.json").read_text())
eng = HybridEngine(OnnxBackend(WEIGHTS / f"{a.name}_int8.onnx", meta), HybridConfig())
P = []
for i in idx:
    img = np.asarray(Image.open(ROOT / "data/prepared" / te.path[i]).convert("RGB"))
    roi = crop(img, dets[i]["box"])
    if needs_clahe(illumination_stats(roi)):
        roi = apply_clahe(roi, clip_limit=2.5)
    x = np.stack([eng._prep(roi), eng._prep(np.ascontiguousarray(roi[:, ::-1]))])
    P.append(softmax(eng.backend.logits(x) / eng.backend.temperature).mean(0))
P = np.stack(P); y = te.label.values[idx]
V = np.load(ROOT / f"data/features/views_{a.name}_test.npz")
g = V["gate"][idx][:, None]
F = 0.5 * (np.where(g, V["crop_c"][idx], V["crop"][idx]) + np.where(g, V["crop_flip_c"][idx], V["crop_flip"][idx]))
acc = lambda p: (round(float((p.argmax(1) == y).mean() * 100), 2), round(float((np.argsort(-p, 1)[:, :5] == y[:, None]).any(1).mean() * 100), 2))
res = {"name": a.name, "n": int(len(idx)), "int8": acc(P), "fp32_same_images": acc(F),
       "argmax_agreement": round(float((P.argmax(1) == F.argmax(1)).mean() * 100), 2)}
(WEIGHTS / f"int8_subset_{a.name}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res))
