# AGENTS.md — Agent & Contributor Operations Manual

> **Mission**: Build an intelligent, fine-grained vehicle recognition and analysis system (Make, Model, Year across 196 categories) integrating Classical Computer Vision + Deep Learning (YOLO detection + Fine-Grained Classification + Hybrid Fusion) with an in-browser web app (ONNX Runtime Web + OpenCV.js) and a FastAPI server, plus quantified ablation analysis for Intra-College Hackathon Evaluation.

---

## 1. Project & Team Identity
- **Problem Statement ID**: P5 (Intelligent Fine-Grained Vehicle Recognition and Analysis System)
- **Team ID**: `T20`
- **Team Members & USNs**:
  - `1RUA24CSE0214` - Chakradhar K (Deep Learning & Kaggle Training)
  - `1RUA24CSE0219` - Sreeram Lagisetty (Team Lead & Full-Stack Integration)
  - `1RUA24CSE0271` - Nagarjuna VV (Classical Computer Vision & Preprocessing)
  - `1RUA24CSE0241` - Manoj R (Vehicle Detection & Localization)
  - `1RUA24CSE0232` - Snehan (Optimization, Failure Analysis & Benchmarking)
- **Dataset**: Stanford Cars Dataset (196 classes, 8,144 train / 8,041 test images)

---

## 2. Hackathon Rubric & Zero-Tolerance Constraints (100 Points)

| Criteria | Max Points | Mandatory Requirements to Win Top Band |
| :--- | :--- | :--- |
| **Demo and Viva** | **25 pts** | Clear 3-minute pitch on live URL; handles judge's live unseen image; each member explains their exact component; explains 3 failure cases. |
| **Evaluation Metrics** | **25 pts** | Top-1 & Top-5 accuracy, IoU, mAP, model size & inference time, robustness tests, confidence calibration. |
| **Implementation** | **20 pts** | Core features work on unseen inputs; proper train/val split; clean modular code; robust error handling for non-car or corrupted images. |
| **UI/UX & Deployment** | **15 pts** | **Public URL works on first click from phone & laptop**; responds in <10s; clear loading & error states; one-command run or Dockerfile. |
| **Project Documentation** | **15 pts** | Architecture diagram matches code; design doc complete; README reproduces results; `eval_P5_T20.md` strictly formatted. |

### ⚠️ Instant Penalties (Do NOT violate):
- **No live URL** $\rightarrow$ UI/UX score capped at `0/15`.
- **No GitHub link** $\rightarrow$ Implementation score capped at `0/20`.
- **Missing Architecture Diagram or Design Doc** $\rightarrow$ Documentation capped at `5/15`.
- **Late submission** $\rightarrow$ Minus 5 points per 15 minutes beyond deadline.

---

## 3. Mandatory Evaluator Output File: `eval_P5_T20.md`
The instruction sheet strictly mandates this exact filename and structure:
- **Line 1 must contain all member USNs comma-separated**:
  `1RUA24CSE0214, 1RUA24CSE0219, 1RUA24CSE0271, 1RUA24CSE0241, 1RUA24CSE0232`
- Followed by the official metrics table in exact order:
  1. Classification performance (Top-1 and Top-5 accuracy)
  2. Object localization (Intersection over Union - IoU)
  3. Object detection (Mean Average Precision - mAP)
  4. Model efficiency (Model size and inference time)
  5. Robustness (Performance under varying illumination/angles)
  6. Confidence (Prediction confidence and threshold analysis)

---

## 4. Architectural Blueprint (as built)

```
Browser (web/, ONNX Runtime Web + OpenCV.js — the deployed app)   |   FastAPI /api/predict (src/hybrid/engine.py)
      │
      ├─► YOLOv8n (ONNX) — car/bus/truck boxes, class-aware NMS
      ├─► Vehicle selection — primary = max(area×conf); others ≥48 px & ≥1 % frame, de-duplicated, ≤6
      ├─► Per vehicle: ROI crop (+5 %) → illumination gate (LAB L mean<70 or std<35) → CLAHE(2.5, 8×8)
      ├─► ConvNeXt classifier (INT8 in browser / FP32 ONNX on server), flip-TTA on the primary vehicle
      ├─► Temperature-calibrated softmax (T fitted on val) → top-5 + status (ok / low_confidence / no_vehicle)
      ├─► [server only] optional log-linear fusion with classical HOG+LBP+HSV → PCA → LogReg head (val-tuned weight)
      └─► UI: numbered boxes, vehicle list, top-5, classical stages (CLAHE, bilateral, Canny, contours), benchmarks
```

Models: `convnext_tiny.fb_in22k_ft_in1k` @224 (default, "Visionary Lite" in browser / "Pro" on server) and
`convnext_small.fb_in22k_ft_in1k` @320 ("Visionary Pro" in browser). Both fine-tuned locally (RTX 3050) with
`scripts/train.py` on GT-box crops with random padding.

---

## 5. Repository Directory Layout

```text
Visionary/
├── README.md · AGENTS.md · eval_P5_T20.md (generated) · Dockerfile · requirements*.txt
├── app/main.py                 # FastAPI: serves web/ (+COOP/COEP) and /api/predict, /api/models, /api/benchmarks
├── web/                        # Deployed in-browser app (HF static Space VISIONALPHA/visionary)
│   ├── index.html · app.js · pipeline.js (JS port of the engine) · styles.css · coi-serviceworker.js
│   ├── vendor/ (onnxruntime-web, opencv.js) · models/ (ONNX + models.json) · data/ · samples/
│   └── test/parity.mjs         # runs pipeline.js in Node for JS-vs-Python parity
├── src/
│   ├── classical/              # CLAHE, bilateral, Canny, contours, HSV/LBP, geometry
│   ├── detection/              # YOLO wrapper, ROI crop, IoU, mAP
│   ├── classification/         # ConvNeXt wrapper (strict weight loading), 196-class metadata
│   └── hybrid/                 # engine.py (pipeline, gate, fusion, multi-vehicle), classical_head.py
├── scripts/                    # prepare_data · train · train_classical · export_onnx · evaluate ·
│                               # benchmark_latency · write_report · build_web · check_web_parity
├── weights/                    # checkpoints, ONNX (FP32/INT8), models.json, eval_*.json (not in git if large)
├── docs/                       # design_document.md · failure_cases.md · failures/*.jpg
├── notebooks/stanford_cars_training_kaggle.ipynb   # runs the same scripts on Kaggle/Colab
└── tests/                      # 49 tests: classical, detection, classification, API (bad input, multi-vehicle)
```

---

## 6. Engineering & Git Workflow Rules for All Agents

1. **Keep `AGENTS.md` and `eval_P5_T20.md` Synchronized**: Whenever a module is finished, mark it complete in the Roadmap Checklist below.
2. **Conventional Commits**: Format every git commit as:
   - `feat(module): description`
   - `fix(module): description`
   - `docs(name): description`
   - `perf(module): description`
3. **Handle Bad & Unseen Input Gracefully**:
   - What if a user uploads a photo of a cat or an empty road? The detector must report *"No vehicle detected"*, not crash with an index error.
   - What if image is corrupted or zero bytes? Catch errors and display a clean alert.
4. **No Secrets**: Never commit tokens, API keys, or private paths.
5. **No Team Members Table in `README.md`**: Do NOT add the team members table (USNs, names, roles) to `README.md`. Team USNs must only reside in `eval_P5_T20.md` (mandatory evaluation header) and `AGENTS.md`.

---

## 7. Master Hackathon Roadmap & Checklist

- [x] Phase 1 — Scaffolding, team identity, eval file header
- [x] Phase 2 — Classical CV (CLAHE, bilateral, Canny, contours, HSV/LBP, geometry)
- [x] Phase 3 — YOLOv8 detection, ROI crop, IoU / mAP@50 / mAP@50:95
- [x] Phase 4 — Fine-tuned ConvNeXt-Tiny @224 and ConvNeXt-Small @320; temperature calibration
- [x] Phase 5 — Hybrid: illumination gate, flip-TTA, classical head + val-tuned log-linear fusion, low-confidence handling, multi-vehicle
- [x] Phase 6 — ONNX export, dynamic INT8 quantisation, CPU latency benchmark (2 threads)
- [x] Phase 7 — In-browser app (ONNX Runtime Web + OpenCV.js), JS/Python parity test, deployed to HF static Space; FastAPI serves the same app
- [ ] Phase 8 — Docs: README, design doc, failure cases, eval file, pitch & viva notes; push to GitHub

