"""
Package the in-browser app (web/): copy INT8 classifiers + YOLO ONNX, write
web/models/models.json (sizes for the download progress bar) and
web/data/eval_results.json (benchmarks + browser-config accuracy).

The browser runs: YOLO ROI + adaptive CLAHE gate + flip TTA + INT8 classifier
(no classical head). Its test accuracy is computed from the cached INT8 views.

Usage:
    python scripts/build_web.py --models convnext_tiny_224 [convnext_small_320] --default convnext_tiny_224
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W, WEB = ROOT / "weights", ROOT / "web"


def browser_acc(name):
    p = ROOT / f"data/features/views_{name}_int8_test.npz"
    if not p.exists():
        sub = W / f"int8_subset_{name}.json"  # stratified-subset INT8 check
        if sub.exists():
            r = json.loads(sub.read_text())
            return r["int8"][0], r["int8"][1]
        return None
    V = np.load(p)
    y = pd.read_csv(ROOT / "data/prepared/test.csv").label.values
    g = V["gate"][:, None]
    prob = 0.5 * (np.where(g, V["crop_c"], V["crop"]) + np.where(g, V["crop_flip_c"], V["crop_flip"]))
    top = np.argsort(-prob, 1)[:, :5]
    return round(float((top[:, 0] == y).mean() * 100), 2), round(float((top == y[:, None]).any(1).mean() * 100), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["convnext_tiny_224"])
    ap.add_argument("--default", default=None)
    args = ap.parse_args()
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(W / "yolov8n.onnx", WEB / "models/yolov8n.onnx")

    models, accs = {}, {}
    names = {"convnext_tiny_224": ("Visionary Lite", "ConvNeXt-Tiny @224 · INT8 · fastest"),
             "convnext_small_320": ("Visionary Pro", "ConvNeXt-Small @320 · INT8 · most accurate")}
    for key in args.models:
        src = W / f"{key}_int8.onnx"
        dst = WEB / "models" / src.name
        shutil.copy(src, dst)
        meta = json.loads(src.with_suffix(".json").read_text())
        acc = browser_acc(key)
        accs[key] = acc
        nm, desc = names.get(key, (key, ""))
        models[key] = {"name": nm, "desc": desc, "file": f"models/{src.name}", "size": dst.stat().st_size,
                       "imgSize": meta["img_size"], "temperature": meta["temperature"],
                       "top1": acc[0] if acc else None}
    default = args.default or max(models, key=lambda k: models[k]["top1"] or 0)
    reg = {"default": default, "models": models,
           "assets": {"opencv": (WEB / "vendor/opencv.js").stat().st_size,
                      "wasm": (WEB / "vendor/ort-wasm-simd-threaded.wasm").stat().st_size,
                      "yolo": (WEB / "models/yolov8n.onnx").stat().st_size}}
    (WEB / "models/models.json").write_text(json.dumps(reg, indent=1))

    # Benchmarks panel shows the most accurate model's full evaluation (same as eval_P5_T20.md)
    best = max(models, key=lambda k: models[k]["top1"] or 0)
    ev = json.loads((W / f"eval_{best}.json").read_text())
    par = json.loads((W / "web_parity.json").read_text()) if (W / "web_parity.json").exists() else {}
    if accs.get(default):
        ev["browser"] = {"top1": accs[default][0], "top5": accs[default][1],
                         "parity": par.get("top1_agreement_js_vs_python"), "model": models[default]["name"]}
    (WEB / "data/eval_results.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps({"default": default, **{k: v["top1"] for k, v in models.items()}}))


if __name__ == "__main__":
    main()
