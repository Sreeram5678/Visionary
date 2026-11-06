"""
Fine-tune an ImageNet-pretrained backbone on Stanford Cars (196 classes).

Key choices (explainable in viva):
  * Transfer learning: start from ImageNet-22k pretrained ConvNeXt; only the
    196-way head is new. Converges in ~10-15 epochs instead of hundreds.
  * Train on bounding-box crops with random padding jitter, because at
    inference time YOLO hands the classifier a (slightly imperfect) crop.
    A fraction of samples use the full image so the model still works if
    the detector misses.
  * Augmentation: flip, TrivialAugment, RandomErasing (simulates occlusion).
  * Label smoothing + AdamW + warmup/cosine LR, mixed precision (AMP).
  * Proper split: train / val (10% of official train, stratified) / test
    (official 8,041 test images, never used for model selection).
  * Post-hoc temperature scaling fitted on val for calibrated confidence.

Usage:
    python scripts/train.py --backbone convnext_tiny.fb_in22k_ft_in1k --img 224 --epochs 15 --name convnext_tiny_224
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification.model import VehicleClassifier  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def jitter_crop(img: Image.Image, box, train: bool, full_prob: float):
    """Crop around the GT box; in training add random padding/shift."""
    if train and random.random() < full_prob:
        return img
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if train:
        pad = [random.uniform(0.0, 0.15) for _ in range(4)]
    else:
        pad = [0.05] * 4
    x1 = max(0, x1 - pad[0] * bw)
    y1 = max(0, y1 - pad[1] * bh)
    x2 = min(W, x2 + pad[2] * bw)
    y2 = min(H, y2 + pad[3] * bh)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img.crop((int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))))


class CarsDataset(Dataset):
    def __init__(self, csv, root, img_size, train, use_crop=True, full_prob=0.15):
        self.df = pd.read_csv(csv)
        self.root = Path(root)
        self.train = train
        self.use_crop = use_crop
        self.full_prob = full_prob
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r.path).convert("RGB")
        if self.use_crop:
            img = jitter_crop(img, (r.x1, r.y1, r.x2, r.y2), self.train, self.full_prob)
        return self.tf(img), int(r.label)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def topk_acc(logits, labels, ks=(1, 5)):
    top = logits.topk(max(ks), dim=1).indices
    hit = top.eq(labels.view(-1, 1))
    return {f"top{k}": hit[:, :k].any(1).float().mean().item() * 100 for k in ks}


def fit_temperature(logits, labels):
    """Grid-search the softmax temperature that minimises NLL on validation."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 3.01, 0.05):
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 3), best_nll


def ece(logits, labels, n_bins=15):
    probs = logits.softmax(1)
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/prepared"))
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr-mult", type=float, default=5.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="convnext_tiny_224")
    ap.add_argument("--out", default=str(ROOT / "weights"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="skip training; calibrate+test the saved checkpoint")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.name}_log.jsonl"
    if not args.eval_only:
        log_path.write_text("")

    data = Path(args.data)
    train_ds = CarsDataset(data / "train.csv", data, args.img, train=True)
    val_ds = CarsDataset(data / "val.csv", data, args.img, train=False)
    test_ds = CarsDataset(data / "test.csv", data, args.img, train=False)
    test_full_ds = CarsDataset(data / "test.csv", data, args.img, train=False, use_crop=False)
    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, shuffle=False, **dl_kw)

    model = VehicleClassifier(backbone_name=args.backbone, num_classes=196, pretrained=True).to(device)
    if args.grad_ckpt and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    params = [
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * args.head_lr_mult},
    ]
    opt = torch.optim.AdamW(params, weight_decay=args.wd)
    steps_per_epoch = len(train_dl) // args.accum
    total = steps_per_epoch * args.epochs
    warmup = steps_per_epoch  # 1 epoch warmup
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        if step < warmup:
            f = (step + 1) / warmup
        else:
            p = (step - warmup) / max(1, total - warmup)
            f = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    best_val, step = -1.0, 0
    ckpt_path = out / f"{args.name}.pt"
    print(f"[{args.name}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"steps/epoch={steps_per_epoch} device={device}", flush=True)

    for epoch in range(0 if args.eval_only else args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(x), y) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                set_lr(step)
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            run_loss += loss.item() * args.accum * x.size(0); seen += x.size(0)
        vl, vy = collect_logits(model, val_dl, device)
        va = topk_acc(vl, vy)
        rec = {"epoch": epoch + 1, "train_loss": round(run_loss / seen, 4),
               "val_top1": round(va["top1"], 2), "val_top5": round(va["top5"], 2),
               "epoch_sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if va["top1"] > best_val:
            best_val = va["top1"]
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                        "img_size": args.img, "num_classes": 196, "epoch": epoch + 1}, ckpt_path)

    # ---- Final: reload best, calibrate on val, evaluate on held-out test ----
    # Release training workers first: on Windows every worker is a full process
    # (~650 MB), and stacking 4 persistent loaders exhausts 16 GB of RAM.
    del train_dl, val_dl
    import gc; gc.collect()
    ev_kw = dict(num_workers=min(4, args.workers), pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, shuffle=False, **ev_kw)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    vl, vy = collect_logits(model, val_dl, device)
    T, _ = fit_temperature(vl, vy)
    test_dl = DataLoader(test_ds, batch_size=args.bs * 2, shuffle=False, **ev_kw)
    test_full_dl = DataLoader(test_full_ds, batch_size=args.bs * 2, shuffle=False, **ev_kw)
    tl, ty = collect_logits(model, test_dl, device)
    tfl, tfy = collect_logits(model, test_full_dl, device)
    res = {
        "name": args.name, "backbone": args.backbone, "img_size": args.img,
        "best_epoch": ck["epoch"], "temperature": T,
        "val": topk_acc(vl, vy),
        "test_gt_crop": topk_acc(tl, ty),
        "test_full_image": topk_acc(tfl, tfy),
        "test_ece_before_T": ece(tl, ty), "test_ece_after_T": ece(tl / T, ty),
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "args": {**vars(args), "data": "data/prepared", "out": "weights"},  # no local paths
    }
    ck["temperature"] = T
    ck["metrics"] = res
    ck["state_dict"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
    torch.save(ck, ckpt_path)  # fp16 storage halves file size
    res["file_MB"] = os.path.getsize(ckpt_path) / 2**20
    (out / f"{args.name}_metrics.json").write_text(json.dumps(res, indent=2))
    print("FINAL", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
