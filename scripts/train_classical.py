"""
Train the classical-only baseline: HOG + LBP + HSV descriptors on the vehicle
ROI -> StandardScaler -> PCA -> multinomial Logistic Regression.

Descriptors are cached to data/features/*.npz so evaluation can reuse them.

Usage:
    python scripts/train_classical.py
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/prepared"
FEAT = ROOT / "data/features"


def _desc(args):
    from src.hybrid.classical_head import classical_descriptor
    path, box = args
    img = np.asarray(Image.open(DATA / path).convert("RGB"))
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - int(0.05 * bw)), max(0, y1 - int(0.05 * bh))
    x2, y2 = min(W, x2 + int(0.05 * bw)), min(H, y2 + int(0.05 * bh))
    roi = img[y1:y2, x1:x2] if (x2 - x1 > 8 and y2 - y1 > 8) else img
    return classical_descriptor(roi)


def features(split: str, workers: int = 6):
    FEAT.mkdir(parents=True, exist_ok=True)
    cache = FEAT / f"classical_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    df = pd.read_csv(DATA / f"{split}.csv")
    jobs = [(r.path, (r.x1, r.y1, r.x2, r.y2)) for r in df.itertuples()]
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        X = np.stack(list(ex.map(_desc, jobs, chunksize=32))).astype(np.float32)
    y = df.label.values
    np.savez_compressed(cache, X=X, y=y)
    print(f"{split}: {X.shape} in {time.time()-t:.0f}s", flush=True)
    return X, y


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = features("train")
    Xva, yva = features("val")
    Xte, yte = features("test")

    def acc(p, y):
        top = np.argsort(-p, 1)[:, :5]
        return float((top[:, 0] == y).mean() * 100), float((top == y[:, None]).any(1).mean() * 100)

    best = None
    for C in [0.01, 0.05, 0.2]:
        pipe = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                             LogisticRegression(C=C, max_iter=600))
        t = time.time()
        pipe.fit(Xtr, ytr)
        v1, v5 = acc(pipe.predict_proba(Xva), yva)
        print(f"C={C}: val top1={v1:.2f} top5={v5:.2f} ({time.time()-t:.0f}s)", flush=True)
        if best is None or v1 > best[0]:
            best = (v1, C, pipe)

    _, C, pipe = best
    # Keep the model fitted on TRAIN only: the validation split is later used to
    # tune the hybrid fusion weight, so it must stay out-of-sample (no leakage).
    t1, t5 = acc(pipe.predict_proba(Xte), yte)
    out = ROOT / "weights/classical_head.joblib"
    joblib.dump(pipe, out, compress=3)
    res = {"C": C, "test_top1": t1, "test_top5": t5, "feature_dim": int(Xtr.shape[1]),
           "file_MB": out.stat().st_size / 2**20}
    (ROOT / "weights/classical_metrics.json").write_text(json.dumps(res, indent=2))
    print("FINAL", res)


if __name__ == "__main__":
    main()
