"""
Full evaluation on the held-out Stanford Cars test set (8,041 images).

Produces:
  * Detection: mean IoU of the primary YOLO box vs GT, detection rate,
    mAP@50 and mAP@50:95 (single class "car").
  * Classification Top-1/Top-5 for classical-only, DL-only, hybrid and INT8.
  * Ablation of every hybrid component (each added one at a time).
  * Robustness under synthetic low light, over-exposure, blur, noise,
    occlusion, viewpoint warp and low resolution.
  * Confidence analysis: avg confidence, low-confidence flag rate, threshold
    sweep (coverage vs accuracy), ECE.
  * CPU latency + model size per variant.
Hybrid fusion weights (w_full, w_classical) are tuned on the VAL split only.

Outputs weights/eval_results.json, weights/models.json and eval_P5_T20.md.

Usage:
    python scripts/evaluate.py --ckpt weights/convnext_tiny_224.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical.preprocessing import apply_clahe  # noqa: E402
from src.detection.metrics import compute_iou, compute_map  # noqa: E402
from src.hybrid.classical_head import ClassicalHead, classical_descriptor  # noqa: E402
from src.hybrid.engine import (WEIGHTS, HybridConfig, HybridEngine, OnnxBackend,  # noqa: E402
                               TorchBackend, illumination_stats, loglinear_fusion, needs_clahe,
                               softmax)

DATA = ROOT / "data/prepared"
CACHE = ROOT / "data/features"
USNS = "1RUA24CSE0214, 1RUA24CSE0219, 1RUA24CSE0271, 1RUA24CSE0241, 1RUA24CSE0232"


def load(path):
    return np.asarray(Image.open(DATA / path).convert("RGB"))


def crop(img, box, pad=0.05):
    if box is None:
        return img
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    x2, y2 = min(W, int(x2 + pad * bw)), min(H, int(y2 + pad * bh))
    return img[y1:y2, x1:x2] if (x2 - x1 > 8 and y2 - y1 > 8) else img


def topk(p, y):
    t = np.argsort(-p, 1)[:, :5]
    return round(float((t[:, 0] == y).mean() * 100), 2), round(float((t == y[:, None]).any(1).mean() * 100), 2)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(e * 100), 2)


# ---------------------------------------------------------------- detection --
def run_yolo(engine, df, tag, images=None):
    cache = CACHE / f"yolo_{tag}.json"
    if cache.exists() and images is None:
        return json.loads(cache.read_text())
    out = []
    for i, r in enumerate(df.itertuples()):
        img = images[i] if images is not None else load(r.path)
        d = engine.detect(img)
        out.append({"box": d.box, "conf": d.conf, "all": [[*b["box"], b["conf"]] for b in d.all_boxes]})
    if images is None:
        cache.write_text(json.dumps(out))
    return out


def detection_metrics(df, dets):
    gt_mask = (df.bbox_source == "gt").values
    ious, preds, gts = [], [], []
    for (r, d, g) in zip(df.itertuples(), dets, gt_mask):
        if not g:
            continue
        gt = [r.x1, r.y1, r.x2, r.y2]
        ious.append(compute_iou(d["box"], gt) if d["box"] else 0.0)
        preds.append(d["all"])
        gts.append([gt])
    maps = [compute_map(preds, gts, iou_threshold=t) for t in np.arange(0.5, 0.96, 0.05)]
    maps = [m["mAP"] if isinstance(m, dict) else m for m in maps]
    ious = np.array(ious)
    return {"n": int(gt_mask.sum()), "mean_iou": round(float(ious.mean()), 4),
            "median_iou": round(float(np.median(ious)), 4),
            "iou_ge_0.5": round(float((ious >= 0.5).mean() * 100), 2),
            "recall": round(float(np.mean([d["box"] is not None for d in dets]) * 100), 2),
            "map50": round(float(maps[0]), 4), "map50_95": round(float(np.mean(maps)), 4)}


# ------------------------------------------------------------------- views --
VIEWS = ["full", "crop", "crop_flip", "full_c", "crop_c", "crop_flip_c", "gt_crop"]


def make_views(img, det_box, gt_box):
    roi = crop(img, det_box)
    roi_c = apply_clahe(roi, clip_limit=2.5)
    full_c = apply_clahe(img, clip_limit=2.5)
    return {"full": img, "crop": roi, "crop_flip": roi[:, ::-1], "full_c": full_c,
            "crop_c": roi_c, "crop_flip_c": roi_c[:, ::-1], "gt_crop": crop(img, gt_box)}, \
        needs_clahe(illumination_stats(roi))


def view_probs(engine, df, dets, tag, images=None, bs=64, views=None):
    views = views or VIEWS
    cache = CACHE / f"views_{tag}.npz"
    if cache.exists() and images is None:
        d = np.load(cache)
        return {k: d[k] for k in views if k in d}, d["gate"]
    acc = {k: [] for k in views}
    gate, buf = [], {k: [] for k in views}
    T = engine.backend.temperature

    def flush():
        for k in views:
            if buf[k]:
                acc[k].append(softmax(engine.backend.logits(np.stack(buf[k])) / T))
                buf[k] = []

    for i, (r, d) in enumerate(zip(df.itertuples(), dets)):
        img = images[i] if images is not None else load(r.path)
        v, g = make_views(img, d["box"], (r.x1, r.y1, r.x2, r.y2))
        gate.append(g)
        for k in views:
            buf[k].append(engine._prep(np.ascontiguousarray(v[k])))
        if len(buf[views[0]]) >= bs:
            flush()
    flush()
    out = {k: np.concatenate(acc[k]).astype(np.float32) for k in views}
    gate = np.array(gate)
    if images is None:
        np.savez_compressed(cache, gate=gate, **out)
    return out, gate


def classical_probs(head, df, dets, tag, images=None):
    cache = CACHE / f"classical_probs_{tag}.npy"
    if cache.exists() and images is None:
        return np.load(cache)
    P = []
    for i, (r, d) in enumerate(zip(df.itertuples(), dets)):
        img = images[i] if images is not None else load(r.path)
        P.append(head.pipe.predict_proba(classical_descriptor(crop(img, d["box"]))[None])[0])
    P = np.stack(P).astype(np.float32)
    if images is None:
        np.save(cache, P)
    return P


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def compose(V, gate, pc, w_full=0.0, w_cls=0.0, flip=False, use_gate=False, crop_on=True, fusion_t=1.0):
    def pick(a, b):
        return np.where(gate[:, None], V[b], V[a]) if use_gate else V[a]
    if not crop_on:
        p = pick("full", "full_c")
    else:
        p = pick("crop", "crop_c")
        if flip:
            p = 0.5 * (p + pick("crop_flip", "crop_flip_c"))
        if w_full > 0:
            p = (1 - w_full) * p + w_full * pick("full", "full_c")
    if w_cls > 0:
        p = loglinear_fusion(p, pc, w_cls, fusion_t)
    return p


# --------------------------------------------------------------- robustness --
def corrupt(img, kind, rng):
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255
    if kind == "low_light":
        f = (f ** 2.2) * 0.35
        f = f + rng.normal(0, 0.02, f.shape)
    elif kind == "over_exposure":
        f = np.clip(f * 1.9 + 0.15, 0, 1)
    elif kind == "motion_blur":
        k = np.zeros((15, 15), np.float32); k[7, :] = 1 / 15
        return cv2.filter2D(img, -1, k)
    elif kind == "gaussian_noise":
        f = f + rng.normal(0, 0.08, f.shape)
    elif kind == "occlusion":
        out = img.copy()
        ow, oh = int(w * 0.35), int(h * 0.45)
        x, y = rng.integers(0, w - ow), rng.integers(h // 4, max(h // 4 + 1, h - oh))
        out[y:y + oh, x:x + ow] = rng.integers(0, 255, 3)
        return out
    elif kind == "viewpoint":
        d = 0.18
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * d, h * d * 0.6], [w, 0], [w, h], [w * d, h * (1 - d * 0.6)]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif kind == "low_resolution":
        s = 80 / max(h, w)
        small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ latency --
def cpu_latency(spec_backend, cfg, head, df, n=25):
    eng = HybridEngine(spec_backend, cfg, classical=head)
    eng.detect(np.zeros((320, 320, 3), np.uint8))
    rows = df.sample(n, random_state=1)
    imgs = [load(p) for p in rows.path]
    eng.analyze(imgs[0])
    ts = []
    for im in imgs:
        r = eng.analyze(im)
        ts.append(r["timings"])
    return {k: round(float(np.mean([t.get(k, 0) for t in ts])), 1) for k in ts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(WEIGHTS / "convnext_tiny_224.pt"))
    ap.add_argument("--int8", default=None)
    ap.add_argument("--robust-n", type=int, default=1000)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-int8", action="store_true", help="skip the (slow, CPU) INT8 test pass")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    name = ckpt.stem
    int8 = Path(args.int8) if args.int8 else ckpt.with_name(name + "_int8.onnx")

    te = pd.read_csv(DATA / "test.csv")
    va = pd.read_csv(DATA / "val.csv")
    yte, yva = te.label.values, va.label.values

    gpu = TorchBackend(ckpt, device="cuda")
    head = ClassicalHead(str(WEIGHTS / "classical_head.joblib"))
    eng = HybridEngine(gpu, HybridConfig(), classical=head)

    print("YOLO on val/test…", flush=True)
    dva = run_yolo(eng, va, "val")
    dte = run_yolo(eng, te, "test")
    det = detection_metrics(te, dte)
    print("detection", det, flush=True)

    print("classifier views…", flush=True)
    Vva, gva = view_probs(eng, va, dva, f"{name}_val")
    Vte, gte = view_probs(eng, te, dte, f"{name}_test")
    print("classical probs…", flush=True)
    Pva = classical_probs(head, va, dva, "val") if head.available else np.zeros_like(Vva["full"])
    Pte = classical_probs(head, te, dte, "test") if head.available else np.zeros_like(Vte["full"])

    # ---- tune fusion weights on VAL only ----
    # Objective: val Top-1, ties broken by lower val NLL (smaller weights first).
    best = None
    for wf in np.arange(0, 0.55, 0.05):
        for wc in ([0, 0.05, 0.1, 0.15, 0.2, 0.3] if head.available else [0]):
            p = compose(Vva, gva, Pva, wf, wc, True, True)
            key = (topk(p, yva)[0], -nll(p, yva))
            if best is None or key > best[0]:
                best = (key, round(float(wf), 2), wc)
    _, w_full, w_cls = best
    fusion_t = 1.0
    if w_cls > 0:  # re-calibrate after fusion (product of experts sharpens)
        fusion_t = float(min(np.arange(0.6, 1.61, 0.05),
                             key=lambda t: nll(compose(Vva, gva, Pva, w_full, w_cls, True, True, fusion_t=t), yva)))
        fusion_t = round(fusion_t, 2)
    print(f"tuned on val: w_full={w_full} w_classical={w_cls} fusion_t={fusion_t}", flush=True)
    FT = dict(fusion_t=fusion_t)

    # ---- ablation on TEST ----
    steps = [
        ("A. Classical only (YOLO ROI + HOG/LBP/HSV + LogReg)", None),
        ("B. Deep learning only (full image)", dict(crop_on=False)),
        ("C. B + adaptive CLAHE gate", dict(crop_on=False, use_gate=True)),
        ("D. YOLO ROI crop + deep learning", dict()),
        ("E. D + horizontal-flip TTA", dict(flip=True)),
        ("F. E + full-frame context fusion", dict(flip=True, w_full=w_full)),
        ("G. F + adaptive CLAHE gate", dict(flip=True, w_full=w_full, use_gate=True)),
        ("H. G + classical late fusion (full hybrid)", dict(flip=True, w_full=w_full, use_gate=True, w_cls=w_cls, **FT)),
    ]
    abl, prev, P = [], None, {}
    for label, kw in steps:
        p = Pte if kw is None else compose(Vte, gte, Pte, **kw)
        P[label[0]] = p
        t1, t5 = topk(p, yte)
        abl.append({"name": label, "top1": t1, "top5": t5,
                    "delta": "" if prev is None or label[0] in "AB" else f"{t1 - prev:+.2f}"})
        prev = t1
    upper = topk(Vte["gt_crop"], yte)
    hybrid_p = P["H"]

    # ---- INT8 on test (hybrid config) ----
    int8_res = None
    if int8.exists() and not args.skip_int8:
        print("INT8 ONNX on test…", flush=True)
        meta = json.loads(int8.with_suffix(".json").read_text())
        ob = OnnxBackend(int8, meta)
        eng8 = HybridEngine(ob, HybridConfig(), classical=head)
        V8, g8 = view_probs(eng8, te, dte, f"{name}_int8_test",
                             views=["crop", "crop_flip", "crop_c", "crop_flip_c"] + (["full", "full_c"] if w_full > 0 else []))
        p8 = compose(V8, g8, Pte, flip=True, w_full=w_full, use_gate=True, w_cls=w_cls, **FT)
        int8_res = {"top1": topk(p8, yte)[0], "top5": topk(p8, yte)[1], "size_mb": round(ob.size_mb, 1)}

    # ---- robustness on a fixed stratified subset ----
    print("robustness…", flush=True)
    rng = np.random.default_rng(0)
    sub = te.groupby("label").sample(n=max(1, args.robust_n // 196), random_state=0).reset_index(drop=True)
    ysub = sub.label.values
    conds = ["clean", "low_light", "over_exposure", "motion_blur", "gaussian_noise",
             "occlusion", "viewpoint", "low_resolution"]
    rob_rows = []
    for c in conds:
        imgs = [load(p) if c == "clean" else corrupt(load(p), c, rng) for p in sub.path]
        dd = run_yolo(eng, sub, f"rob_{c}", images=imgs)
        V, g = view_probs(eng, sub, dd, f"rob_{c}", images=imgs)
        pc = classical_probs(head, sub, dd, f"rob_{c}", images=imgs) if head.available else np.zeros_like(V["full"])
        vals = [topk(pc, ysub)[0],
                topk(compose(V, g, pc, crop_on=False), ysub)[0],
                topk(compose(V, g, pc, flip=True, w_full=w_full, use_gate=True, w_cls=w_cls, **FT), ysub)[0],
                round(float(np.mean([x["box"] is not None for x in dd]) * 100), 1)]
        rob_rows.append({"condition": c.replace("_", " "), "values": vals})
        print(c, vals, flush=True)

    # ---- confidence analysis (hybrid, test) ----
    conf = hybrid_p.max(1)
    correct = hybrid_p.argmax(1) == yte
    sweep = []
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = conf >= th
        sweep.append({"threshold": th, "coverage": round(float(m.mean() * 100), 2),
                      "accuracy_on_accepted": round(float(correct[m].mean() * 100), 2) if m.any() else None})
    confidence = {
        "avg_conf_all": round(float(conf.mean() * 100), 2),
        "avg_conf_correct": round(float(conf[correct].mean() * 100), 2),
        "avg_conf_wrong": round(float(conf[~correct].mean() * 100), 2),
        "low_conf_rate": round(float((conf < 0.5).mean() * 100), 2),
        "acc_when_flagged": round(float(correct[conf < 0.5].mean() * 100), 2),
        "acc_when_confident": round(float(correct[conf >= 0.5].mean() * 100), 2),
        "ece_hybrid": ece(hybrid_p, yte),
        "temperature": gpu.temperature,
        "sweep": sweep,
    }

    # ---- failure mining: most confused pairs ----
    pred = hybrid_p.argmax(1)
    names = json.loads((DATA / "class_names.json").read_text())
    pairs = pd.Series([f"{names[a]} -> {names[b]}" for a, b in zip(yte[~correct], pred[~correct])]).value_counts()
    per_cls = pd.Series(correct).groupby(yte).mean().sort_values()
    failures = {"top_confusions": pairs.head(10).to_dict(),
                "worst_classes": {names[i]: round(float(v * 100), 1) for i, v in per_cls.head(8).items()},
                "n_errors": int((~correct).sum())}
    # save a few concrete failure examples for docs
    wrong_idx = np.where(~correct)[0]
    examples = []
    for i in wrong_idx[np.argsort(-conf[wrong_idx])][:6]:
        examples.append({"path": te.path.iloc[i], "true": names[yte[i]], "pred": names[pred[i]],
                         "conf": round(float(conf[i] * 100), 1),
                         "iou": round(compute_iou(dte[i]["box"], [te.x1.iloc[i], te.y1.iloc[i], te.x2.iloc[i], te.y2.iloc[i]]), 3) if dte[i]["box"] else 0})
    failures["high_conf_examples"] = examples

    # ---- latency (CPU, deployment target) ----
    lat = {}
    if not args.skip_latency:
        print("CPU latency…", flush=True)
        import torch
        torch.set_num_threads(4)
        cpu_b = TorchBackend(ckpt, device="cpu")
        cfg = HybridConfig(w_full=w_full, w_classical=w_cls, fusion_t=fusion_t)
        lat["fp32"] = cpu_latency(cpu_b, cfg, head, te)
        lat["dl_only"] = cpu_latency(cpu_b, HybridConfig(w_full=0, use_flip_tta=False, use_clahe_gate=False,
                                                        use_detector=False), None, te)
        if int8.exists():
            lat["int8"] = cpu_latency(OnnxBackend(int8, json.loads(int8.with_suffix(".json").read_text())), cfg, head, te)
        t = time.perf_counter()
        for p in te.path.iloc[:25]:
            head.pipe.predict_proba(classical_descriptor(load(p))[None])
        lat["classical_ms"] = round((time.perf_counter() - t) * 40, 1)
        print(lat, flush=True)

    size_fp = round(gpu.size_mb, 1)
    size_cls = round((WEIGHTS / "classical_head.joblib").stat().st_size / 2**20, 1) if head.available else None
    size_yolo = round((WEIGHTS / "yolov8n.pt").stat().st_size / 2**20, 1)
    comparison = [
        {"name": "Classical only", "top1": abl[0]["top1"], "top5": abl[0]["top5"],
         "latency_ms": lat.get("classical_ms"), "size_mb": size_cls},
        {"name": "Deep learning only", "top1": abl[1]["top1"], "top5": abl[1]["top5"],
         "latency_ms": lat.get("dl_only", {}).get("total_ms"), "size_mb": size_fp},
        {"name": "Hybrid (FP16 weights, FP32 compute)", "top1": abl[-1]["top1"], "top5": abl[-1]["top5"],
         "latency_ms": lat.get("fp32", {}).get("total_ms"), "size_mb": round(size_fp + size_yolo + (size_cls or 0), 1)},
    ]
    if int8_res:
        comparison.append({"name": "Hybrid INT8 (ONNX Runtime)", "top1": int8_res["top1"], "top5": int8_res["top5"],
                           "latency_ms": lat.get("int8", {}).get("total_ms"),
                           "size_mb": round(int8_res["size_mb"] + size_yolo + (size_cls or 0), 1)})

    results = {"n_test": int(len(te)), "model": name, "params_m": round(gpu.params_m, 2),
               "hybrid_weights": {"w_full": w_full, "w_classical": w_cls, "fusion_t": fusion_t},
               "comparison": comparison, "ablation": abl, "upper_bound_gt_crop": upper,
               "detection": det, "int8": int8_res, "latency": lat, "confidence": confidence,
               "robustness": {"methods": ["Classical", "DL only", "Hybrid", "Detect %"], "rows": rob_rows,
                              "n": int(len(sub))},
               "failures": failures}
    out = WEIGHTS / f"eval_{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
