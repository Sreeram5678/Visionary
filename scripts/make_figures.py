"""
Generate the report / documentation figures into docs/figures/ from the real
pipeline and evaluation outputs (nothing is hand-drawn or mocked).

Usage: python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classical.edges import detect_edges_canny, extract_contours  # noqa: E402
from src.classical.preprocessing import apply_clahe, denoise_bilateral  # noqa: E402
from src.hybrid.engine import available_registry, build_engine  # noqa: E402

OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
W = ROOT / "weights"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1e1c", "#6f6b64", "#e6e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
VEH = [(201, 100, 66), (42, 120, 214), (27, 175, 122), (237, 161, 0), (232, 123, 164), (0, 131, 0)]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def annotate(rgb, res):
    img = rgb.copy()
    h, w = img.shape[:2]
    th = max(2, w // 220)
    fs = max(0.45, w / 1100)
    for v in reversed(res["vehicles"]):
        if not v["box"]:
            continue
        x1, y1, x2, y2 = [int(round(c)) for c in v["box"]]
        col = VEH[(v["index"] - 1) % len(VEH)]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th + (1 if v["index"] == 1 else 0))
        p = v["prediction"]
        multi = len(res["vehicles"]) > 1
        label = (f"{'#1 ' if multi else ''}{p['year']} {p['make']} {p['model']} {v['confidence']:.0f}%"
                 if v["index"] == 1 else f"#{v['index']}")
        (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly = max(0, y1 - tht - bl - 6)
        cv2.rectangle(img, (x1, ly), (min(w - 1, x1 + tw + 8), ly + tht + bl + 6), col, -1)
        cv2.putText(img, label, (x1 + 4, ly + tht + 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fig_samples(engine):
    items = [("web/samples/ferrari.jpg", "Single car"), ("web/samples/bmw_dark.jpg", "Dark image (CLAHE gate on)"),
             ("data/prepared/images/test/test_00631.jpg", "Several cars")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (p, title) in zip(axes, items):
        rgb = load(ROOT / p)
        r = engine.analyze(rgb)
        ax.imshow(annotate(rgb, r))
        ax.set_title(f"{title}\n{r['prediction']['name']} · {r['confidence']:.1f}%"
                     + (f" · {len(r['vehicles'])} vehicles" if len(r["vehicles"]) > 1 else ""), fontsize=9.5, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_samples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_stages():
    rgb = load(ROOT / "web/samples/bmw_dark.jpg")
    eng_det = engine_any()
    r = eng_det.analyze(rgb)
    roi = r["_roi"]
    clahe = apply_clahe(roi, clip_limit=2.5)
    den = denoise_bilateral(clahe)
    edges = detect_edges_canny(den)
    _, cont = extract_contours(edges, min_area=50.0)
    panels = [(roi, f"1. YOLO ROI\n(mean L {r['illumination']['mean_luma']:.0f} → dark)"),
              (clahe, "2. CLAHE on LAB L\n(clip 2.5, 8×8)"), (den, "3. Bilateral filter\n(d 9, σ 75)"),
              (edges, "4. Canny\n(median ± 33 %)"), (cont, "5. Contours\n(closing, area ≥ 50)")]
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.9))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=9, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stages.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


_ENG = {}


def engine_any():
    if "e" not in _ENG:
        reg = available_registry()
        _ENG["e"] = build_engine(reg["default"], reg, device="cpu")
    return _ENG["e"]


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=INK)


def fig_comparison():
    lite, pro = json.loads((W / "eval_convnext_tiny_224.json").read_text()), json.loads((W / "eval_convnext_small_320.json").read_text())
    cats = ["Classical only", "Deep learning only", "Hybrid"]
    val = lambda E: [E["comparison"][0]["top1"], E["comparison"][1]["top1"],
                     next(r for r in E["comparison"] if r["name"].startswith("Hybrid ("))["top1"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    b1 = ax.bar(x - 0.2, val(lite), 0.38, color=BLUE, label="Lite (ConvNeXt-Tiny @224)")
    b2 = ax.bar(x + 0.2, val(pro), 0.38, color=ORANGE, label="Pro (ConvNeXt-Small @320)")
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(x, cats); ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Test-set Top-1 accuracy (8,041 unseen images)", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_comparison.png", dpi=170); plt.close(fig)


def fig_robustness():
    E = json.loads((W / "eval_convnext_small_320.json").read_text())
    rows = E["robustness"]["rows"]
    names = [r["condition"] for r in rows]
    dl = [r["values"][1] for r in rows]; hy = [r["values"][2] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(y + 0.2, dl, 0.38, color=BLUE, label="Deep learning only")
    ax.barh(y - 0.2, hy, 0.38, color=ORANGE, label="Hybrid")
    for yy, a, b in zip(y, dl, hy):
        ax.text(a + 0.4, yy + 0.2, f"{a:.1f}", va="center", fontsize=8, color=INK)
        ax.text(b + 0.4, yy - 0.2, f"{b:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names); ax.set_xlim(50, 100); ax.set_xlabel("Top-1 accuracy (%)")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=9)
    ax.set_title(f"Robustness — Pro model, {E['robustness']['n']} test images per condition", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_robustness.png", dpi=170); plt.close(fig)


def fig_confidence():
    E = json.loads((W / "eval_convnext_small_320.json").read_text())
    s = E["confidence"]["sweep"]
    t = [r["threshold"] for r in s]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(t, [r["coverage"] for r in s], color=BLUE, lw=2, marker="o", ms=5, label="Coverage (answers kept)")
    ax.plot(t, [r["accuracy_on_accepted"] for r in s], color=ORANGE, lw=2, marker="o", ms=5, label="Accuracy on kept answers")
    ax.axvline(0.5, color=MUTED, lw=1, ls="--"); ax.text(0.51, 99.9, "threshold used (0.5)", fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("%"); ax.set_ylim(88, 100.5)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Confidence threshold trade-off (Pro, test set)", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_confidence.png", dpi=170); plt.close(fig)


def fig_failures():
    items = [("docs/failures/bentley_2012_as_2007.jpg", "True: Bentley Continental GT 2012", "Pred: … GT 2007 (facelift year)"),
             ("docs/failures/ferrari_coupe_as_convertible.jpg", "True: Ferrari 458 Italia Coupe", "Pred: 458 Italia Convertible"),
             ("docs/failures/corolla_as_camry.jpg", "True: Toyota Corolla Sedan 2012", "Pred: Toyota Camry Sedan 2012")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (p, a, b) in zip(axes, items):
        ax.imshow(load(ROOT / p)); ax.axis("off"); ax.set_title(f"{a}\n{b}", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig_failures.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col + "22", ec=col, lw=1.6))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.8, color=MUTED, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    y = 2.3
    box(0.1, y, 1.6, 1.3, "Input photo", "upload / drag / paste\nEXIF-rotated, ≤1600 px", MUTED)
    box(2.1, y, 1.9, 1.3, "Detection", "YOLOv8n (ONNX)\ncar / bus / truck\nNMS · all vehicles", BLUE)
    box(4.4, y, 2.0, 1.3, "Classical gate", "LAB L mean<70 or std<35\n→ CLAHE (clip 2.5, 8×8)", AQUA)
    box(6.8, y, 2.1, 1.3, "Classifier", "fine-grained ConvNeXt (Tiny / Small)\nROI + flipped ROI\nINT8 / FP32 ONNX", ORANGE)
    box(9.3, y, 1.7, 1.3, "Calibration", "softmax(z / T)\nT fitted on val\n+ log-linear fusion", ORANGE)
    box(11.4, y, 1.5, 1.3, "Decision", "≥ 50 % → answer\n< 50 % → warn\nno box → no vehicle", MUTED)
    for a, b in [(1.7, 2.1), (4.0, 4.4), (6.4, 6.8), (8.9, 9.3), (11.0, 11.4)]:
        arrow(a, y + 0.65, b, y + 0.65)
    box(4.4, 0.3, 2.0, 1.25, "Classical stages", "bilateral · Canny (auto)\ncontours (UI panels)", AQUA)
    box(6.8, 0.3, 2.1, 1.25, "Classical head", "HOG + LBP + HSV\n→ PCA 256 → LogReg", AQUA)
    box(9.3, 0.3, 3.6, 1.25, "Outputs (browser UI / REST API)", "numbered boxes · make/model/year · top-5\nconfidence · stage images · benchmarks", MUTED)
    arrow(5.4, y, 5.4, 1.55); arrow(7.85, 1.55, 9.6, y)
    arrow(12.15, y, 12.15, 1.55)
    ax.text(0.1, 4.1, "Same pipeline in two runtimes: web/pipeline.js (ONNX Runtime Web + OpenCV.js, deployed) and "
            "src/hybrid/engine.py (FastAPI /api/predict) — parity verified.", fontsize=8.6, color=MUTED)
    fig.savefig(OUT / "fig_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_comparison(); fig_robustness(); fig_confidence(); fig_failures()
    fig_stages(); fig_samples(engine_any())
    print(sorted(p.name for p in OUT.glob("*.png")))
