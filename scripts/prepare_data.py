"""
Prepare Stanford Cars for fast training.

Reads the parquet mirror (naufalso/stanford_cars: image + bbox + label + name),
writes every image to disk as JPEG (longest side capped) and produces CSV
manifests with scaled bounding boxes, plus a stratified train/val split.

Usage:
    python scripts/prepare_data.py --src data/stanford_cars --out data/prepared
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

MAX_SIDE = 640


def export_split(files, out_dir: Path, split: str):
    img_dir = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for f in files:
        table = pq.read_table(f)
        df = table.to_pandas()
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{split}:{Path(f).name}"):
            img_field = r["image_path"]
            raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = img.size
            scale = min(1.0, MAX_SIDE / max(w, h))
            if scale < 1.0:
                img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            name = f"{split}_{idx:05d}.jpg"
            if not (img_dir / name).exists():
                img.save(img_dir / name, quality=93)
            rows.append({
                "bbox_source": "gt",
                "path": f"images/{split}/{name}",
                "label": int(r["label"]),
                "car_name": r["car_name"],
                "x1": float(r["x_min"]) * scale,
                "y1": float(r["y_min"]) * scale,
                "x2": float(r["x_max"]) * scale,
                "y2": float(r["y_max"]) * scale,
                "width": img.size[0],
                "height": img.size[1],
            })
            idx += 1
    return pd.DataFrame(rows)


def export_missing_class(extra_dir: Path, out_dir: Path, split: str, label0: int, name: str):
    """
    The bbox mirror drops class 174 ("Ram C/V Cargo Van Minivan 2012"; the '/'
    broke its export). Recover its images from tanganke/stanford_cars (labels,
    no boxes) and pseudo-label the box with YOLOv8 (bbox_source='yolo').
    These rows are excluded from IoU/mAP since they have no ground-truth box.
    """
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    img_dir = out_dir / "images" / split
    rows = []
    files = sorted(extra_dir.glob(f"{split}-*.parquet"))
    k = 0
    for f in files:
        df = pq.read_table(f).to_pandas()
        df = df[df["label"] == label0]
        for _, r in df.iterrows():
            raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = img.size
            scale = min(1.0, MAX_SIDE / max(w, h))
            if scale < 1.0:
                img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            fname = f"{split}_extra_{k:03d}.jpg"
            img.save(img_dir / fname, quality=93)
            res = yolo.predict(np.asarray(img)[:, :, ::-1], classes=[2, 5, 7], conf=0.2, verbose=False)[0]
            W, H = img.size
            box = (0, 0, W, H)
            if res.boxes is not None and len(res.boxes):
                xy = res.boxes.xyxy.cpu().numpy()
                a = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
                box = tuple(float(v) for v in xy[int(a.argmax())])
            rows.append({"bbox_source": "yolo", "path": f"images/{split}/{fname}", "label": label0 + 1,
                         "car_name": name, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                         "width": W, "height": H})
            k += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/stanford_cars")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--extra", default="data/tanganke")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_files = sorted(str(p) for p in src.glob("train-*.parquet"))
    test_files = sorted(str(p) for p in src.glob("test-*.parquet"))
    assert train_files and test_files, f"parquet files not found in {src}"

    train_df = export_split(train_files, out, "train")
    test_df = export_split(test_files, out, "test")

    extra = Path(args.extra)
    present = set(train_df.label)
    for lab1 in sorted(set(range(1, 197)) - present):
        if not list(extra.glob("train-*.parquet")):
            break
        nm = json.loads(pq.read_schema(next(extra.glob("train-*.parquet"))).metadata[b"huggingface"]) \
            ["info"]["features"]["label"]["names"][lab1 - 1]
        print(f"recovering missing class {lab1} ({nm}) from {extra}")
        train_df = pd.concat([train_df, export_missing_class(extra, out, "train", lab1 - 1, nm)], ignore_index=True)
        test_df = pd.concat([test_df, export_missing_class(extra, out, "test", lab1 - 1, nm)], ignore_index=True)

    # Labels in the mirror may be 1-based (devkit convention); normalise to 0..195.
    lo = min(train_df.label.min(), test_df.label.min())
    train_df["label"] -= lo
    test_df["label"] -= lo
    n_cls = train_df.label.nunique()
    assert n_cls == 196, f"expected 196 classes, got {n_cls}"

    # Stratified train/val split from the official train set.
    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(train_df), dtype=bool)
    for _, grp in train_df.groupby("label"):
        k = max(1, int(round(len(grp) * args.val_frac)))
        val_mask[rng.choice(grp.index.values, size=k, replace=False)] = True
    train_df["split"] = np.where(val_mask, "val", "train")
    test_df["split"] = "test"

    train_df[train_df.split == "train"].to_csv(out / "train.csv", index=False)
    train_df[train_df.split == "val"].to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    names = (train_df.drop_duplicates("label").sort_values("label")["car_name"].tolist())
    with open(out / "class_names.json", "w") as fh:
        json.dump(names, fh, indent=1)

    print(f"train={int((~val_mask).sum())} val={int(val_mask.sum())} test={len(test_df)} classes={n_cls}")


if __name__ == "__main__":
    main()
