# Visionary — 3-minute demo script & viva preparation

Live app: **https://visionalpha-visionary.static.hf.space** (open it once before
judging so the models are cached; first visit downloads ~66 MB).

---

## 1. Three-minute demo script

| Time | Who | What to do / say |
|---|---|---|
| 0:00–0:20 | Lead | "Visionary identifies a car's **make, model and year** across 196 classes. It runs **entirely in the browser** — no server, photos never leave the phone." Show the empty page. |
| 0:20–0:50 | Lead | Click **🚗 Try a sample** → Ferrari. Point at the box, the name, **100 % confidence**, the top-5 bars. |
| 0:50–1:20 | Classical CV | Click **Try a sample** again → the **dark BMW**. Point at the badge **"☀ CLAHE applied"**. Open **"How Visionary got this answer"** → walk through ROI → CLAHE → bilateral → Canny → contours. |
| 1:20–1:50 | Detection | Upload a **photo with several cars** (parking lot). "YOLOv8 finds every vehicle; each one is cropped and classified — numbered boxes, and the list marks uncertain ones." |
| 1:50–2:20 | DL / Optimisation | Open **📊 Benchmarks**: classical 19 % → deep learning 94.0 % → **hybrid 94.7 %** Top-1 on 8,041 unseen test images (Pro model); INT8 is 3.5× smaller with no accuracy loss. "The phone runs Lite by default (94.1 %) for speed; Pro is one tap away." |
| 2:20–2:50 | Failure analysis | Hand the judge the phone: "Try any car." If it's a modern/Indian car → "It says *low confidence* and shows alternatives — the dataset only has US cars up to 2012. Below 50 % confidence we're right 47 % of the time, above it 95 %." |
| 2:50–3:00 | Lead | "Code, eval file, design doc, failure cases are on GitHub. Thank you." |

**Backup if Wi-Fi is bad:** run `uvicorn app.main:app` on the laptop and open
`http://127.0.0.1:7860` (same app, models served locally).

---

## 2. Key numbers to remember

| Metric | Value |
|---|---|
| Test images (never trained on) | 8,041 (official Stanford Cars test split) |
| Top-1 / Top-5 — hybrid **Pro** (ConvNeXt-Small @320) | **94.7 % / 99.3 %** (INT8 in browser: 94.4 %) |
| Top-1 / Top-5 — hybrid **Lite** (ConvNeXt-Tiny @224) | 94.0 % / 99.1 % (INT8 in browser: 94.1 %) |
| Top-1 — deep learning only (full image) | 94.0 % Pro · 92.4 % Lite |
| Top-1 — classical only (HOG+LBP+HSV) | 19.0 % (random = 0.5 %) |
| Biggest single gain (ablation) | YOLO ROI crop: **+1.7 pts** (Lite), +0.5 pts (Pro) — bigger models need less help |
| Detection | mean IoU 0.93, detection rate 99 %, mAP@50 0.78 |
| Calibration | ECE 19.4 % → **1.9 %** after temperature scaling (T = 0.55) |
| INT8 | 3.3× smaller model, −0.05 pts Top-1 |
| Browser parity | JS = Python accuracy (93.33 % on the same 300 images), 98 % same answer |
| Latency | ~2–3 s per photo in a laptop browser; ~0.14–0.2 s on server CPU (2 threads) |

---

## 3. Who explains what (each member owns one part)

### Chakradhar K — Deep learning & training
* **Model:** ConvNeXt-Tiny (28 M params), ImageNet-22k pretrained, new 196-class head. Why ConvNeXt: modern CNN, strong transfer, fast on CPU, exports cleanly to ONNX.
* **Training trick that matters:** trained on **ground-truth box crops with random 0–15 % padding**, because at inference YOLO gives an imperfect crop; 15 % full images so it still works if detection fails.
* Augmentation: flip, TrivialAugment, RandomErasing (simulates occlusion). AdamW, warm-up + cosine LR, label smoothing 0.1, mixed precision, 15 epochs (~12 min on an RTX 3050).
* **Split:** train 7,356 / val 788 (10 % stratified) / test 8,041. Best epoch and calibration chosen on **val only**.
* **Calibration:** label smoothing makes the model under-confident; temperature T = 0.55 fitted on val fixes it (ECE 19 % → 2 %).

### Sreeram Lagisetty — Integration, app & deployment
* Two front doors, one pipeline: **browser app** (`web/`, ONNX Runtime Web + OpenCV.js) and **FastAPI** (`/api/predict`). `web/pipeline.js` is a line-by-line port of `src/hybrid/engine.py`; `scripts/check_web_parity.py` proves they agree.
* Why in-browser: free hosting (HF static Space), no cold start, privacy, scales to any number of judges.
* Threads need cross-origin isolation → `coi-serviceworker` on static hosting; FastAPI sets COOP/COEP headers itself.
* Bad input: empty file, non-image, < 32 px, > 15 MB, no car → clean messages, never a crash (tested).

### Nagarjuna VV — Classical computer vision
* **CLAHE** on the LAB **L** channel (not RGB) so colours don't shift; clip 2.5, 8×8 tiles.
* **Adaptive gate:** CLAHE only when the crop is dark (mean L < 70) or flat (std < 35). Always applying it would change images the network was trained on — ablation shows the gate is neutral on clean images and helps under low light/blur.
* Visual stages: bilateral filter (edge-preserving denoise) → Canny with **automatic thresholds** (median ± 33 %) → morphological closing → external contours.
* **Classical baseline:** HOG (shape) + LBP (texture) + HSV histogram → PCA 256 → logistic regression = 19 % Top-1. Colour alone can't tell models apart (any model comes in any colour); HOG carries most of the signal.

### Manoj R — Detection & localisation
* YOLOv8n (6 MB, COCO-pretrained), only car/bus/truck classes, conf ≥ 0.25.
* Primary vehicle = largest **area × confidence** box (the foreground car, not a distant one).
* Multi-vehicle: every other box ≥ 48 px and ≥ 1 % of the image, duplicates removed (IoU > 0.5 or 80 % inside another — e.g. the same pickup labelled car *and* truck), max 6.
* IoU vs ground truth: mean 0.93, 98.8 % of images ≥ 0.5. mAP@50 = 0.78 — lower than IoU suggests because Stanford Cars labels only the main car, so correct detections of background cars count as false positives.

### Snehan — Optimisation, evaluation & failures
* **Optimisation:** FP16 checkpoint (53 MB), ONNX export, **dynamic INT8** quantisation (MatMul/Gemm) → 32 MB, −0.05 pts. ONNX Runtime is 5.7× faster than PyTorch on CPU.
* **Ablation:** every component added one at a time on the same cached detections (table in `eval_P5_T20.md`).
* **Robustness:** 7 synthetic conditions; hybrid beats DL-only in all except very low resolution.
* **Failures:** see `docs/failure_cases.md` — facelift years, trims, roof-up convertibles, low resolution, out-of-distribution cars.

---

## 4. Likely viva questions

**Q: Why is the classical-only accuracy so low? Isn't that a failure?**
A: 196 classes that differ in small details; random guessing is 0.5 %, so 19 % shows the features carry real information. It's the baseline the rubric asks for. Classical CV earns its place inside the hybrid — the illumination gate, and the ROI the classifier sees.

**Q: Did the classical fusion actually help?**
A: Honestly, not on clean accuracy. On validation the tuned weight looked useful, but we found the classical head had been fitted on train **plus** val (a leak), fixed it, and the clean-test effect became ≈ 0. We report that rather than tuning on the test set. The measurable classical contribution is the CLAHE gate and robustness.

**Q: How do you know you didn't overfit to the test set?**
A: Test images are never used for training, early stopping, temperature or fusion weights — all chosen on the validation split.

**Q: Why ConvNeXt and not YOLO for classification / a ViT?**
A: YOLO is trained on 80 generic classes; fine-grained make/model needs a dedicated classifier on the crop. ConvNeXt-Tiny gives ViT-level transfer accuracy with CNN speed and clean INT8/ONNX export; it runs in a browser.

**Q: What happens with a car from 2020 or an Indian model?**
A: The model can only pick from its 196 classes, so it returns the closest; confidence is usually low and the UI warns. Next step: open-set detection on embeddings.

**Q: Why is mAP lower than IoU?**
A: Only one car per image is annotated; extra correct detections are counted as false positives.

**Q: What would you improve with more time?**
A: Higher resolution / part attention for facelift years; hierarchical make→model→trim output; open-set rejection; fine-tuning YOLO on vehicle boxes; WebGPU for faster phones.
