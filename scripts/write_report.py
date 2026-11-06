"""
Assemble final artefacts from evaluation outputs:
  * eval_P5_T20.md          (mandatory submission file, official metric order)
  * weights/eval_results.json (served to the UI benchmarks panel)
  * weights/models.json      (model registry for the UI model switcher)

Usage:
    python scripts/write_report.py --default convnext_tiny_224 [--extra convnext_small_288]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights"
USNS = "1RUA24CSE0214, 1RUA24CSE0219, 1RUA24CSE0271, 1RUA24CSE0241, 1RUA24CSE0232"
TIER_NAMES = {"convnext_small_320": "Visionary Pro", "convnext_tiny_224": "Visionary Lite"}


def j(p):
    p = W / p
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, suf=""):
    return "–" if v is None else f"{v}{suf}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", default="convnext_tiny_224")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    E = j(f"eval_{args.default}.json")
    tr = j(f"{args.default}_metrics.json")
    cls = j("classical_metrics.json")
    assert E, "run scripts/evaluate.py first"

    # ------------------------------------------------------------ registry --
    models = {}
    hw = E["hybrid_weights"]
    for key in [args.default, *args.extra]:
        e = j(f"eval_{key}.json")
        if not e:
            continue
        h = next(r for r in e["comparison"] if r["name"].startswith("Hybrid ("))
        meta = json.loads((W / f"{key}_int8.json").read_text()) if (W / f"{key}_int8.json").exists() else {}
        backbone = meta.get("backbone", key).split(".")[0].replace("convnext_", "ConvNeXt-").title().replace("Convnext", "ConvNeXt")
        size = meta.get("img_size", "")
        fp32 = W / f"{key}_fp32.onnx"
        hyb = {"w_full": e["hybrid_weights"]["w_full"], "w_classical": e["hybrid_weights"]["w_classical"],
               "fusion_t": e["hybrid_weights"].get("fusion_t", 1.0)}
        tier_name = TIER_NAMES.get(key, key)
        models[key] = {"name": tier_name,
                       "desc": f"{backbone} @{size}px · full precision",
                       "runtime": "onnx" if fp32.exists() else "torch",
                       "file": fp32.name if fp32.exists() else f"{key}.pt", **meta, "hybrid": hyb, "tier": "full",
                       "top1": h["top1"], "latency_ms": h["latency_ms"]}
        if e.get("int8"):
            q = next(r for r in e["comparison"] if "INT8" in r["name"])
            models[f"{key}_int8"] = {"name": f"{tier_name} · INT8",
                                     "desc": f"{backbone} @{size}px · INT8 quantised (faster)",
                                     "runtime": "onnx", "file": f"{key}_int8.onnx", **meta, "hybrid": hyb,
                                     "tier": "lite", "top1": q["top1"], "latency_ms": q["latency_ms"]}
    # default = the most accurate full-precision model
    best = max((k for k, m in models.items() if m["tier"] == "full"), key=lambda k: models[k]["top1"])
    reg = {"default": best, "hybrid": models[best]["hybrid"], "models": models}
    (W / "models.json").write_text(json.dumps(reg, indent=2))
    (W / "eval_results.json").write_text(json.dumps(j(f"eval_{best}.json"), indent=2))
    E = j(f"eval_{best}.json")
    tr = j(f"{best}_metrics.json") or tr

    # ------------------------------------------------------------ markdown --
    C = {r["name"]: r for r in E["comparison"]}
    cl, dl = E["comparison"][0], E["comparison"][1]
    hy = E["comparison"][2]
    q8 = E["comparison"][3] if len(E["comparison"]) > 3 else None
    d = E["detection"]
    conf = E["confidence"]
    lat = E.get("latency", {})
    L = []
    a = L.append
    a(USNS)
    a("")
    a("# Evaluation Results — Problem Statement 5 (P5), Team T20")
    a("")
    a("**Project:** Visionary — Intelligent Fine-Grained Vehicle Recognition and Analysis System  ")
    a(f"**Dataset:** Stanford Cars, 196 classes. Train 7,356 / Val 788 (stratified 10% of official train) / "
      f"**Test {E['n_test']:,} (official test split, never used for training or model selection)**.  ")
    a(f"**Classifier:** `{tr['backbone'] if tr else E['model']}` fine-tuned at {tr['img_size'] if tr else ''} px "
      f"({E['params_m']} M params), temperature-calibrated (T = {conf['temperature']}).  ")
    a("**Detector:** YOLOv8n (COCO-pretrained, car/bus/truck classes).  ")
    a("**Latency:** CPU, 2 threads (emulating the 2-vCPU free hosting tier), median of 40 test images, end-to-end incl. detection.")
    a("")
    a("## 1. Classification performance (Top-1 / Top-5 accuracy)")
    a("")
    a("| Approach | Top-1 (%) | Top-5 (%) |")
    a("|---|---|---|")
    for r in E["comparison"]:
        a(f"| {r['name']} | {r['top1']} | {r['top5']} |")
    a("")
    a(f"Reference upper bound — classifier on ground-truth box crops: Top-1 {E['upper_bound_gt_crop'][0]}%, "
      f"Top-5 {E['upper_bound_gt_crop'][1]}%.")
    if E.get("int8", {}).get("subset_n"):
        q = E["int8"]
        a("")
        a(f"\\* INT8 measured on a class-stratified subset of {q['subset_n']:,} test images (CPU-bound pass); on the "
          f"same images FP32 scores {q['fp32_same_images'][0]}% / {q['fp32_same_images'][1]}% and "
          f"{q['argmax_agreement']}% of predictions are identical.")
    variants = [k for k in [args.default, *args.extra] if j(f"eval_{k}.json")]
    if len(variants) > 1:
        a("")
        a("**Model variants (both evaluated on the full test set; the app's model picker switches between them):**")
        a("")
        a("| Variant | Backbone / input | Params (M) | DL-only Top-1 | Hybrid Top-1 | Hybrid Top-5 | Hybrid CPU latency (ms) |")
        a("|---|---|---|---|---|---|---|")
        for k in variants:
            e = j(f"eval_{k}.json")
            m = j(f"{k}_metrics.json") or {}
            hyb = next(r for r in e["comparison"] if r["name"].startswith("Hybrid ("))
            a(f"| {TIER_NAMES.get(k, k)} | {m.get('backbone', k).split('.')[0]} @ {m.get('img_size', '')} px | "
              f"{e['params_m']} | {e['comparison'][1]['top1']} | {hyb['top1']} | {hyb['top5']} | {fmt(hyb['latency_ms'])} |")
    a("")
    a("## 2. Object localization (Intersection over Union)")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Mean IoU (primary YOLO box vs GT, {d['n']:,} test images) | {d['mean_iou']} |")
    a(f"| Median IoU | {d['median_iou']} |")
    a(f"| Images with IoU ≥ 0.5 | {d['iou_ge_0.5']}% |")
    a(f"| Vehicle detection rate | {d['recall']}% |")
    a("")
    a("## 3. Object detection (Mean Average Precision)")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| mAP@0.50 | {d['map50']} |")
    a(f"| mAP@0.50:0.95 | {d['map50_95']} |")
    a("")
    a("Note: Stanford Cars annotates only the main car per image, so correct detections of background cars count "
      "as false positives; mAP is therefore a conservative lower bound.")
    a("")
    a("## 4. Model efficiency (model size and inference time)")
    a("")
    a("| Variant | Size on disk (MB) | CPU latency / image (ms) |")
    a("|---|---|---|")
    for r in E["comparison"]:
        a(f"| {r['name']} | {fmt(r['size_mb'])} | {fmt(r['latency_ms'])} |")
    if lat.get("fp32"):
        f = lat["fp32"]
        a("")
        n_views = 2 + (1 if hw["w_full"] > 0 else 0)
        head = f" · classical head {f.get('classical_head_ms')} ms" if hw["w_classical"] > 0 else ""
        a(f"Hybrid latency breakdown (CPU): detection {f.get('detection_ms')} ms · classical CV gate "
          f"{f.get('classical_ms')} ms · classifier ({n_views} views: ROI + flip"
          f"{' + full frame' if n_views == 3 else ''}) {f.get('classifier_ms')} ms{head}.")
    if q8:
        a("")
        q = E["int8"]
        # Compare INT8 with FP32 on the *same* images (the subset when INT8 was run on one).
        ref = q["fp32_same_images"][0] if q.get("subset_n") else hy["top1"]
        fp32_mb = round((W / f"{E['model']}_fp32.onnx").stat().st_size / 2**20, 1) if (W / f"{E['model']}_fp32.onnx").exists() else None
        a(f"Optimisation: dynamic INT8 weight quantisation (ONNX Runtime) changes Top-1 by "
          f"{q8['top1'] - ref:+.2f} pts on the same images while shrinking the classifier from "
          f"{fmt(fp32_mb)} MB to {q['size_mb']} MB and cutting CPU latency from {fmt(hy['latency_ms'])} ms to "
          f"{fmt(q8['latency_ms'])} ms.")
    a("")
    a("## 5. Robustness (performance under varying conditions)")
    a("")
    R = E["robustness"]
    a(f"Top-1 accuracy (%) on a fixed class-stratified subset of {R['n']} test images with synthetic corruptions "
      f"(last column: % of images where YOLO found a vehicle).")
    a("")
    a("| Condition | " + " | ".join(R["methods"]) + " |")
    a("|---|" + "---|" * len(R["methods"]))
    for r in R["rows"]:
        a(f"| {r['condition']} | " + " | ".join(str(v) for v in r["values"]) + " |")
    a("")
    a("## 6. Confidence (prediction confidence and threshold analysis)")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Average confidence (all / correct / wrong) | {conf['avg_conf_all']}% / {conf['avg_conf_correct']}% / {conf['avg_conf_wrong']}% |")
    a(f"| Low-confidence flag rate (< 50%) | {conf['low_conf_rate']}% |")
    a(f"| Accuracy when flagged / when confident | {conf['acc_when_flagged']}% / {conf['acc_when_confident']}% |")
    a(f"| Expected Calibration Error (hybrid) | {conf['ece_hybrid']}% |")
    if tr:
        a(f"| ECE before / after temperature scaling (classifier) | {tr['test_ece_before_T']:.2f}% / {tr['test_ece_after_T']:.2f}% |")
    a("")
    a("| Threshold | Coverage (%) | Accuracy on accepted (%) |")
    a("|---|---|---|")
    for s in conf["sweep"]:
        a(f"| {s['threshold']} | {s['coverage']} | {s['accuracy_on_accepted']} |")
    a("")
    a("## 7. Ablation study")
    a("")
    a("Each row adds one component to the row above (A and B are independent baselines). "
      f"Fusion weights tuned on validation only: w_full = {hw['w_full']}, w_classical = {hw['w_classical']}.")
    a("")
    a("| Configuration | Top-1 (%) | Top-5 (%) | Δ Top-1 |")
    a("|---|---|---|---|")
    for r in E["ablation"]:
        a(f"| {r['name']} | {r['top1']} | {r['top5']} | {r['delta'] or '–'} |")
    a("")
    a("## 8. Failure-case analysis")
    a("")
    F = E["failures"]
    a(f"Total errors on test: {F['n_errors']:,} / {E['n_test']:,}. Most frequent confusions (true → predicted):")
    a("")
    a("| Confusion | Count |")
    a("|---|---|")
    for k, v in list(F["top_confusions"].items())[:8]:
        a(f"| {k} | {v} |")
    a("")
    a("Hardest classes (per-class test accuracy):")
    a("")
    for k, v in F["worst_classes"].items():
        a(f"- {k}: {v}%")
    a("")
    a("Detailed root-cause analysis of the failure cases is in `docs/failure_cases.md`.")
    a("")
    if cls:
        a(f"_Classical baseline details: HOG + LBP + HSV histogram ({cls['feature_dim']} dims) → PCA(256) → "
          f"multinomial logistic regression (C = {cls['C']})._")
    (ROOT / "eval_P5_T20.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote eval_P5_T20.md, weights/models.json, weights/eval_results.json; default =", best)


if __name__ == "__main__":
    main()
