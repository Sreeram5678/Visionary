1RUA24CSE0214, 1RUA24CSE0219, 1RUA24CSE0271, 1RUA24CSE0241, 1RUA24CSE0232

# Evaluation Results — Problem Statement 5 (P5), Team T20

**Project:** Visionary — Intelligent Fine-Grained Vehicle Recognition and Analysis System  
**Dataset:** Stanford Cars, 196 classes. Train 7,356 / Val 788 (stratified 10% of official train) / **Test 8,041 (official test split, never used for training or model selection)**.  
**Classifier:** `convnext_small.fb_in22k_ft_in1k` fine-tuned at 320 px (49.61 M params), temperature-calibrated (T = 0.6).  
**Detector:** YOLOv8n (COCO-pretrained, car/bus/truck classes).  
**Latency:** CPU, 2 threads (emulating the 2-vCPU free hosting tier), median of 40 test images, end-to-end incl. detection.

## 1. Classification performance (Top-1 / Top-5 accuracy)

| Approach | Top-1 (%) | Top-5 (%) |
|---|---|---|
| Classical only | 19.02 | 41.49 |
| Deep learning only | 94.03 | 99.28 |
| Hybrid (FP32, ONNX Runtime) | 94.73 | 99.3 |
| Hybrid INT8 (ONNX Runtime; 1,960-image subset*) | 94.44 | 99.44 |

Reference upper bound — classifier on ground-truth box crops: Top-1 94.89%, Top-5 99.37%.

\* INT8 measured on a class-stratified subset of 1,960 test images (CPU-bound pass); on the same images FP32 scores 94.44% / 99.44% and 99.9% of predictions are identical.

**Model variants (both evaluated on the full test set; the app's model picker switches between them):**

| Variant | Backbone / input | Params (M) | DL-only Top-1 | Hybrid Top-1 | Hybrid Top-5 | Hybrid CPU latency (ms) |
|---|---|---|---|---|---|---|
| Visionary Pro | convnext_small @ 320 px | 49.61 | 94.03 | 94.73 | 99.3 | 537.8 |
| Visionary Lite | convnext_tiny @ 224 px | 27.97 | 92.35 | 93.99 | 99.08 | 206.3 |

## 2. Object localization (Intersection over Union)

| Metric | Value |
|---|---|
| Mean IoU (primary YOLO box vs GT, 8,000 test images) | 0.9302 |
| Median IoU | 0.9522 |
| Images with IoU ≥ 0.5 | 98.76% |
| Vehicle detection rate | 99.03% |

## 3. Object detection (Mean Average Precision)

| Metric | Value |
|---|---|
| mAP@0.50 | 0.7769 |
| mAP@0.50:0.95 | 0.7128 |

Note: Stanford Cars annotates only the main car per image, so correct detections of background cars count as false positives; mAP is therefore a conservative lower bound.

## 4. Model efficiency (model size and inference time)

| Variant | Size on disk (MB) | CPU latency / image (ms) |
|---|---|---|
| Classical only | 2.3 | 18.7 |
| Deep learning only | 94.7 | 248.2 |
| Hybrid (FP32, ONNX Runtime) | 198.0 | 537.8 |
| Hybrid INT8 (ONNX Runtime; 1,960-image subset*) | 62.8 | 287.2 |

Hybrid latency breakdown (CPU): detection 62.4 ms · classical CV gate 0.9 ms · classifier (2 views: ROI + flip) 468.4 ms.

Optimisation: dynamic INT8 weight quantisation (ONNX Runtime) changes Top-1 by +0.00 pts on the same images while shrinking the classifier from 189.4 MB to 54.3 MB and cutting CPU latency from 537.8 ms to 287.2 ms.

## 5. Robustness (performance under varying conditions)

Top-1 accuracy (%) on a fixed class-stratified subset of 980 test images with synthetic corruptions (last column: % of images where YOLO found a vehicle).

| Condition | Classical | DL only | Hybrid | Detect % |
|---|---|---|---|---|
| clean | 17.65 | 93.37 | 94.49 | 99.0 |
| low light | 6.02 | 90.71 | 89.08 | 89.0 |
| over exposure | 10.1 | 91.53 | 92.35 | 96.3 |
| motion blur | 10.92 | 84.8 | 85.41 | 97.9 |
| gaussian noise | 9.18 | 92.76 | 92.55 | 86.8 |
| occlusion | 3.98 | 89.08 | 89.29 | 98.4 |
| viewpoint | 17.14 | 92.55 | 94.29 | 98.8 |
| low resolution | 9.29 | 61.63 | 61.73 | 89.6 |

## 6. Confidence (prediction confidence and threshold analysis)

| Metric | Value |
|---|---|
| Average confidence (all / correct / wrong) | 96.12% / 97.39% / 73.31% |
| Low-confidence flag rate (< 50%) | 1.77% |
| Accuracy when flagged / when confident | 39.44% / 95.72% |
| Expected Calibration Error (hybrid) | 1.77% |
| ECE before / after temperature scaling (classifier) | 16.76% / 1.94% |

| Threshold | Coverage (%) | Accuracy on accepted (%) |
|---|---|---|
| 0.3 | 99.43 | 95.13 |
| 0.4 | 98.98 | 95.36 |
| 0.5 | 98.23 | 95.72 |
| 0.6 | 96.89 | 96.25 |
| 0.7 | 95.26 | 96.76 |
| 0.8 | 93.27 | 97.23 |
| 0.9 | 90.45 | 97.75 |

## 7. Ablation study

Each row adds one component to the row above (A and B are independent baselines). Fusion weights tuned on validation only: w_full = 0.0, w_classical = 0.

| Configuration | Top-1 (%) | Top-5 (%) | Δ Top-1 |
|---|---|---|---|
| A. Classical only (YOLO ROI + HOG/LBP/HSV + LogReg) | 19.02 | 41.49 | – |
| B. Deep learning only (full image) | 94.03 | 99.28 | – |
| C. B + adaptive CLAHE gate | 94.08 | 99.27 | +0.05 |
| D. YOLO ROI crop + deep learning | 94.58 | 99.29 | +0.50 |
| E. D + horizontal-flip TTA | 94.73 | 99.32 | +0.15 |
| F. E + full-frame context fusion | 94.73 | 99.32 | +0.00 |
| G. F + adaptive CLAHE gate | 94.73 | 99.3 | +0.00 |
| H. G + classical late fusion (full hybrid) | 94.73 | 99.3 | +0.00 |

## 8. Failure-case analysis

Total errors on test: 424 / 8,041. Most frequent confusions (true → predicted):

| Confusion | Count |
|---|---|
| Audi S5 Coupe 2012 -> Audi A5 Coupe 2012 | 14 |
| Chevrolet Express Van 2007 -> Chevrolet Express Cargo Van 2007 | 13 |
| Audi TTS Coupe 2012 -> Audi TT Hatchback 2011 | 12 |
| Audi V8 Sedan 1994 -> Audi 100 Sedan 1994 | 12 |
| Bentley Continental GT Coupe 2012 -> Bentley Continental GT Coupe 2007 | 10 |
| Dodge Caliber Wagon 2012 -> Dodge Caliber Wagon 2007 | 10 |
| Suzuki SX4 Sedan 2012 -> Suzuki Aerio Sedan 2007 | 10 |
| Dodge Caliber Wagon 2007 -> Dodge Caliber Wagon 2012 | 10 |

Hardest classes (per-class test accuracy):

- Chevrolet Express Van 2007: 42.9%
- Chevrolet Express Cargo Van 2007: 65.5%
- Audi S5 Coupe 2012: 66.7%
- Audi TTS Coupe 2012: 66.7%
- Bentley Continental GT Coupe 2012: 70.6%
- Audi V8 Sedan 1994: 72.1%
- Audi TT Hatchback 2011: 72.5%
- Audi 100 Sedan 1994: 75.0%

Detailed root-cause analysis of the failure cases is in `docs/failure_cases.md`.

_Classical baseline details: HOG + LBP + HSV histogram (2302 dims) → PCA(256) → multinomial logistic regression (C = 0.01)._
