"""
CPU latency benchmark emulating the deployment target (HF Spaces free tier:
2 vCPU). Reports median and p90 end-to-end latency per variant over N test
images, and patches weights/eval_<name>.json.

Usage:
    python scripts/benchmark_latency.py --name convnext_tiny_224 --threads 2 --n 40
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="convnext_tiny_224")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(args.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(args.threads)

from src.hybrid.classical_head import ClassicalHead, classical_descriptor  # noqa: E402
from src.hybrid.engine import WEIGHTS, HybridConfig, HybridEngine, OnnxBackend, TorchBackend  # noqa: E402

E_PATH = WEIGHTS / f"eval_{args.name}.json"
E = json.loads(E_PATH.read_text())
hw = E["hybrid_weights"]
df = pd.read_csv(ROOT / "data/prepared/test.csv").sample(args.n, random_state=1)
imgs = [np.asarray(Image.open(ROOT / "data/prepared" / p).convert("RGB")) for p in df.path]
head = ClassicalHead(str(WEIGHTS / "classical_head.joblib"))


def bench(engine):
    engine.analyze(imgs[0]); engine.analyze(imgs[1])  # warm-up
    tot, parts = [], []
    for im in imgs:
        r = engine.analyze(im)
        tot.append(r["timings"]["total_ms"]); parts.append(r["timings"])
    out = {k: round(float(np.median([p.get(k, 0) for p in parts])), 1) for k in parts[0]}
    out["total_ms"] = round(float(np.median(tot)), 1)
    out["p90_ms"] = round(float(np.percentile(tot, 90)), 1)
    return out


hy_cfg = HybridConfig(w_full=hw["w_full"], w_classical=hw["w_classical"], fusion_t=hw.get("fusion_t", 1.0))
dl_cfg = HybridConfig(w_full=0, use_flip_tta=False, use_clahe_gate=False, use_detector=False)
tb = TorchBackend(WEIGHTS / f"{args.name}.pt", device="cpu")
lat = {"threads": args.threads, "n_images": args.n}
lat["dl_only"] = bench(HybridEngine(tb, dl_cfg))
lat["torch_fp32"] = bench(HybridEngine(tb, hy_cfg, classical=head))
import onnxruntime as ort  # noqa: E402
meta = json.loads((WEIGHTS / f"{args.name}_int8.json").read_text())
for tag in ("fp32", "int8"):
    p = WEIGHTS / f"{args.name}_{tag}.onnx"
    if p.exists():
        ob = OnnxBackend(p, meta)
        so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
        ob.sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        lat[tag] = bench(HybridEngine(ob, hy_cfg, classical=head))
if "fp32" not in lat:
    lat["fp32"] = lat["torch_fp32"]
t = []
for im in imgs:
    s = time.perf_counter(); head.pipe.predict_proba(classical_descriptor(im)[None]); t.append((time.perf_counter() - s) * 1000)
lat["classical_ms"] = round(float(np.median(t)), 1)

E["latency"] = lat
m = {"Classical only": lat["classical_ms"], "Deep learning only": lat["dl_only"]["total_ms"]}
for r in E["comparison"]:
    if r["name"] in m:
        r["latency_ms"] = m[r["name"]]
    elif "INT8" in r["name"] and "int8" in lat:
        r["latency_ms"] = lat["int8"]["total_ms"]
    elif r["name"].startswith("Hybrid ("):
        r["name"] = "Hybrid (FP32, ONNX Runtime)"
        r["latency_ms"] = lat["fp32"]["total_ms"]
        fp = WEIGHTS / f"{args.name}_fp32.onnx"
        if fp.exists():
            r["size_mb"] = round(fp.stat().st_size / 2**20 + (WEIGHTS / "yolov8n.pt").stat().st_size / 2**20
                                 + (WEIGHTS / "classical_head.joblib").stat().st_size / 2**20, 1)
E_PATH.write_text(json.dumps(E, indent=2))
print(json.dumps(lat, indent=1))
