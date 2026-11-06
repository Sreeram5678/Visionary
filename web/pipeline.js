/*
 * Visionary — in-browser hybrid inference pipeline.
 *
 * A faithful port of src/hybrid/engine.py that runs entirely client-side:
 *   1. YOLOv8n (ONNX) vehicle detection  — Ultralytics letterbox + class-aware NMS
 *   2. ROI crop with 5 % padding           — same integer maths as src/detection/cropper.py
 *   3. Adaptive illumination gate          — LAB lightness mean/std, CLAHE (clip 2.5, 8x8) if dark/flat
 *   4. ConvNeXt classifier (INT8 ONNX)     — bicubic 224 resize, ImageNet norm, ROI + flipped ROI
 *   5. Temperature-calibrated softmax, averaged over views; confidence handling
 * plus the classical-CV visualisation stages (CLAHE, bilateral, Canny, contours).
 *
 * Environment-agnostic: works in the browser and in Node (used by web/test/parity.mjs).
 * Inputs are ImageData-like objects {data: RGBA Uint8ClampedArray, width, height}.
 */

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const VEHICLE = { 2: "car", 5: "bus", 7: "truck" };
const YOLO_SIZE = 640;
const MAX_SIDE = 1600;

export class Pipeline {
  /**
   * @param {object} ort   onnxruntime-web namespace
   * @param {object} cv    initialised OpenCV.js namespace
   * @param {object} cfg   { classes, temperature, imgSize, lowConf, detConf, dark, flat }
   */
  constructor(ort, cv, cfg) {
    this.ort = ort;
    this.cv = cv;
    this.cfg = Object.assign(
      { temperature: 1, imgSize: 224, lowConf: 0.5, detConf: 0.25, nmsIou: 0.7, dark: 70, flat: 35,
        clahe: 2.5, flipTTA: true, clahegate: true },
      cfg
    );
    this.detector = null;
    this.classifier = null;
  }

  async load(yoloBytes, clsBytes, opts = {}) {
    const so = { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...opts };
    this.detector = await this.ort.InferenceSession.create(yoloBytes, so);
    await this.setClassifier(clsBytes, opts);
  }

  async setClassifier(clsBytes, opts = {}, cfgPatch = {}) {
    const so = { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...opts };
    if (this.classifier && this.classifier.release) await this.classifier.release();
    this.classifier = await this.ort.InferenceSession.create(clsBytes, so);
    Object.assign(this.cfg, cfgPatch);
  }

  /* ------------------------------------------------------------------ helpers */
  _rgbMat(img) {
    const cv = this.cv;
    const rgba = cv.matFromImageData(img);
    const rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    rgba.delete();
    return rgb;
  }

  static toImageData(cv, rgbOrGray) {
    const out = new cv.Mat();
    if (rgbOrGray.channels() === 1) cv.cvtColor(rgbOrGray, out, cv.COLOR_GRAY2RGBA);
    else cv.cvtColor(rgbOrGray, out, cv.COLOR_RGB2RGBA);
    const res = { data: new Uint8ClampedArray(out.data), width: out.cols, height: out.rows };
    out.delete();
    return res;
  }

  _capSize(rgb) {
    const cv = this.cv;
    const m = Math.max(rgb.rows, rgb.cols);
    if (m <= MAX_SIDE) return rgb;
    const s = MAX_SIDE / m;
    const out = new cv.Mat();
    cv.resize(rgb, out, new cv.Size(Math.round(rgb.cols * s), Math.round(rgb.rows * s)), 0, 0, cv.INTER_AREA);
    rgb.delete();
    return out;
  }

  /* ---------------------------------------------------------------- detection */
  async detect(rgb) {
    const cv = this.cv;
    const t0 = now();
    const h = rgb.rows, w = rgb.cols;
    const r = Math.min(YOLO_SIZE / h, YOLO_SIZE / w);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const dw = (YOLO_SIZE - nw) / 2, dh = (YOLO_SIZE - nh) / 2;
    const top = Math.round(dh - 0.1), left = Math.round(dw - 0.1);
    const bottom = Math.round(dh + 0.1), right = Math.round(dw + 0.1);
    const resized = new cv.Mat();
    if (nw !== w || nh !== h) cv.resize(rgb, resized, new cv.Size(nw, nh), 0, 0, cv.INTER_LINEAR);
    else rgb.copyTo(resized);
    const padded = new cv.Mat();
    cv.copyMakeBorder(resized, padded, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114, 255));
    resized.delete();

    const N = YOLO_SIZE * YOLO_SIZE;
    const input = new Float32Array(3 * N);
    const d = padded.data;
    for (let i = 0; i < N; i++) {
      input[i] = d[i * 3] / 255;
      input[N + i] = d[i * 3 + 1] / 255;
      input[2 * N + i] = d[i * 3 + 2] / 255;
    }
    padded.delete();
    const feeds = { [this.detector.inputNames[0]]: new this.ort.Tensor("float32", input, [1, 3, YOLO_SIZE, YOLO_SIZE]) };
    const out = (await this.detector.run(feeds))[this.detector.outputNames[0]].data; // [1,84,8400]
    const A = 8400;
    const cands = [];
    for (let a = 0; a < A; a++) {
      let best = -1, bc = -1;
      for (let c = 0; c < 80; c++) {
        const s = out[(4 + c) * A + a];
        if (s > best) { best = s; bc = c; }
      }
      if (best < this.cfg.detConf || !(bc in VEHICLE)) continue;
      const cx = out[a], cy = out[A + a], bw = out[2 * A + a], bh = out[3 * A + a];
      cands.push({ x1: cx - bw / 2, y1: cy - bh / 2, x2: cx + bw / 2, y2: cy + bh / 2, conf: best, cls: bc });
    }
    // Ultralytics: best class per anchor (multi_label=False), class-aware NMS.
    cands.sort((p, q) => q.conf - p.conf);
    const keep = [];
    for (const b of cands) {
      if (keep.some(k => k.cls === b.cls && iou(k, b) > this.cfg.nmsIou)) continue;
      keep.push(b);
      if (keep.length >= 300) break;
    }
    const boxes = keep.map(b => {
      const x1 = clamp((b.x1 - left) / r, 0, w), y1 = clamp((b.y1 - top) / r, 0, h);
      const x2 = clamp((b.x2 - left) / r, 0, w), y2 = clamp((b.y2 - top) / r, 0, h);
      return { box: [x1, y1, x2, y2], conf: b.conf, label: VEHICLE[b.cls] };
    });
    let primary = null;
    for (const b of boxes) {
      const s = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) * b.conf;
      if (!primary || s > primary._s) primary = Object.assign({ _s: s }, b);
    }
    return { boxes, primary, ms: now() - t0 };
  }

  /* --------------------------------------------------------------- classical */
  crop(rgb, box, pad = 0.05) {
    if (!box) return rgb.clone();
    const [x1, y1, x2, y2] = box;
    const bw = x2 - x1, bh = y2 - y1;
    const px = pyRound(bw * pad), py = pyRound(bh * pad);
    const cx1 = Math.max(0, Math.trunc(x1) - px), cy1 = Math.max(0, Math.trunc(y1) - py);
    const cx2 = Math.min(rgb.cols, Math.trunc(x2) + px), cy2 = Math.min(rgb.rows, Math.trunc(y2) + py);
    if (cx2 <= cx1 || cy2 <= cy1) return rgb.clone();
    return rgb.roi(new this.cv.Rect(cx1, cy1, cx2 - cx1, cy2 - cy1)).clone();
  }

  illumination(rgb) {
    const cv = this.cv;
    const lab = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const d = lab.data;
    let s = 0, s2 = 0;
    const n = d.length / 3;
    for (let i = 0; i < d.length; i += 3) { s += d[i]; s2 += d[i] * d[i]; }
    lab.delete();
    const mean = s / n;
    return { mean_luma: mean, contrast: Math.sqrt(Math.max(0, s2 / n - mean * mean)) };
  }

  clahe(rgb, clip = this.cfg.clahe) {
    const cv = this.cv;
    const lab = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const ch = new cv.MatVector();
    cv.split(lab, ch);
    const L = ch.get(0), Leq = new cv.Mat();
    const cl = new cv.CLAHE(clip, new cv.Size(8, 8));
    cl.apply(L, Leq);
    ch.set(0, Leq);
    cv.merge(ch, lab);
    const out = new cv.Mat();
    cv.cvtColor(lab, out, cv.COLOR_Lab2RGB);
    [lab, L, Leq].forEach(m => m.delete());
    ch.delete(); cl.delete();
    return out;
  }

  /** CLAHE -> bilateral -> auto-Canny -> contours, as RGBA images for the UI. */
  stages(roi) {
    const cv = this.cv;
    const clahe = this.clahe(roi);
    const den = new cv.Mat();
    cv.bilateralFilter(clahe, den, 9, 75, 75, cv.BORDER_DEFAULT);
    const gray = new cv.Mat(), blur = new cv.Mat(), edges = new cv.Mat();
    cv.cvtColor(den, gray, cv.COLOR_RGB2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);
    const med = median(blur.data);
    let lo = Math.trunc(Math.max(0, 0.67 * med)), hi = Math.trunc(Math.min(255, 1.33 * med));
    if (lo === hi) { lo = Math.max(0, lo - 30); hi = Math.min(255, hi + 30); }
    cv.Canny(blur, edges, lo, hi);
    const closed = new cv.Mat(), kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(closed, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const vis = cv.Mat.zeros(edges.rows, edges.cols, cv.CV_8UC3);
    let bestI = -1, bestA = -1;
    const keep = new cv.MatVector();
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i), a = cv.contourArea(c);
      if (a >= 50) {
        keep.push_back(c);
        if (a > bestA) { bestA = a; bestI = keep.size() - 1; }
      }
      c.delete();
    }
    if (keep.size() > 0) {
      cv.drawContours(vis, keep, -1, new cv.Scalar(0, 255, 255), 1);
      cv.drawContours(vis, keep, bestI, new cv.Scalar(0, 255, 0), 2);
    }
    const T = m => Pipeline.toImageData(cv, m);
    const res = [
      { name: "Vehicle ROI", desc: "YOLOv8 crop of the primary vehicle", img: T(roi) },
      { name: "CLAHE", desc: "Contrast-limited equalisation on LAB lightness", img: T(clahe) },
      { name: "Bilateral denoise", desc: "Edge-preserving smoothing", img: T(den) },
      { name: "Canny edges", desc: `Auto thresholds ${lo}–${hi} (median ± 33%)`, img: T(edges) },
      { name: "Contours", desc: `${keep.size()} structural contours`, img: T(vis) },
    ];
    [clahe, den, gray, blur, edges, closed, kernel, hier, vis].forEach(m => m.delete());
    contours.delete(); keep.delete();
    return res;
  }

  /* ---------------------------------------------------------- classification */
  _prep(rgb, flip, out, offset) {
    const cv = this.cv, S = this.cfg.imgSize;
    let src = rgb, tmp = null;
    if (flip) { tmp = new cv.Mat(); cv.flip(rgb, tmp, 1); src = tmp; }
    const r = new cv.Mat();
    cv.resize(src, r, new cv.Size(S, S), 0, 0, cv.INTER_CUBIC);
    const d = r.data, N = S * S;
    for (let i = 0; i < N; i++) {
      out[offset + i] = (d[i * 3] / 255 - MEAN[0]) / STD[0];
      out[offset + N + i] = (d[i * 3 + 1] / 255 - MEAN[1]) / STD[1];
      out[offset + 2 * N + i] = (d[i * 3 + 2] / 255 - MEAN[2]) / STD[2];
    }
    r.delete();
    if (tmp) tmp.delete();
  }

  async classify(roi) {
    return (await this.classifyMany([roi]))[0];
  }

  /**
   * One batched forward pass for several ROIs. Flip-TTA is applied to the
   * primary vehicle only (index 0) — secondary vehicles get a single view to
   * keep multi-car scenes fast.
   */
  async classifyMany(rois) {
    const S = this.cfg.imgSize, per = 3 * S * S;
    const views = [];  // [roiIndex, flip]
    rois.forEach((_, j) => {
      views.push([j, false]);
      if (this.cfg.flipTTA && j === 0) views.push([j, true]);
    });
    const buf = new Float32Array(views.length * per);
    views.forEach(([j, flip], v) => this._prep(rois[j], flip, buf, v * per));
    const feeds = { [this.classifier.inputNames[0]]: new this.ort.Tensor("float32", buf, [views.length, 3, S, S]) };
    const logits = (await this.classifier.run(feeds))[this.classifier.outputNames[0]].data;
    const K = logits.length / views.length;
    const out = rois.map(() => new Float64Array(K)), cnt = rois.map(() => 0);
    views.forEach(([j], v) => {
      const row = softmax(logits.subarray(v * K, (v + 1) * K), this.cfg.temperature);
      for (let k = 0; k < K; k++) out[j][k] += row[k];
      cnt[j]++;
    });
    out.forEach((p, j) => { for (let k = 0; k < K; k++) p[k] /= cnt[j]; });
    return out;
  }

  /**
   * Vehicles worth classifying: primary first, then others by area*conf,
   * dropping duplicates (IoU > 0.5 or mostly inside a kept box, e.g. the same
   * pickup reported as both "car" and "truck") and tiny background specks.
   */
  selectVehicles(det, W, H) {
    // Below ~48 px / 1 % of the frame a car is too small to name reliably.
    const minSide = this.cfg.minSide ?? 48, minArea = (this.cfg.minAreaFrac ?? 0.01) * W * H;
    const maxN = this.cfg.maxVehicles ?? 6;
    const area = b => (b[2] - b[0]) * (b[3] - b[1]);
    const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
                            Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const ranked = det.boxes.slice().sort((a, b) => area(b.box) * b.conf - area(a.box) * a.conf);
    const keep = [];
    for (const b of ranked) {
      const w = b.box[2] - b.box[0], h = b.box[3] - b.box[1];
      const isPrimary = det.primary && b.box === det.primary.box;
      if (!isPrimary && (Math.min(w, h) < minSide || w * h < minArea)) continue;
      const dup = keep.some(k => {
        const i = inter(k.box, b.box);
        return i / (area(k.box) + area(b.box) - i) > 0.5 || i / area(b.box) > 0.8;
      });
      if (dup) continue;
      keep.push(b);
      if (keep.length >= maxN) break;
    }
    return keep;
  }

  _status(conf, hasBox) {
    if (!hasBox && conf < this.cfg.lowConf) return "no_vehicle";
    return conf < this.cfg.lowConf ? "low_confidence" : "ok";
  }

  _top5(p) {
    const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 5);
    return order.map(i => ({ class_id: i, ...this.cfg.classes[i], confidence: +(p[i] * 100).toFixed(2) }));
  }

  /* ------------------------------------------------------------ end to end */
  async analyze(img, { withStages = true } = {}) {
    const cv = this.cv;
    const t0 = now();
    const rgb = this._capSize(this._rgbMat(img));
    const det = await this.detect(rgb);
    const box = det.primary ? det.primary.box : null;

    // Every sufficiently large, non-duplicate vehicle (primary first).
    const picked = box ? this.selectVehicles(det, rgb.cols, rgb.rows) : [null];
    let t = now();
    const crops = picked.map(v => {
      const roi = this.crop(rgb, v ? v.box : null);
      const illum = this.illumination(roi);
      const gate = this.cfg.clahegate && (illum.mean_luma < this.cfg.dark || illum.contrast < this.cfg.flat);
      return { roi, illum, gate, roiIn: gate ? this.clahe(roi) : roi };
    });
    const classicalMs = now() - t;

    t = now();
    const probs = await this.classifyMany(crops.map(c => c.roiIn));
    const classifierMs = now() - t;

    const vehicles = picked.map((v, j) => {
      const top5 = this._top5(probs[j]);
      return { index: j + 1, box: v ? v.box : null, det_conf: v ? v.conf : 0, label: v ? v.label : "",
               top5, prediction: top5[0], confidence: top5[0].confidence,
               status: this._status(top5[0].confidence / 100, !!v), clahe_applied: crops[j].gate };
    });
    const roi = crops[0].roi, illum = crops[0].illum, gate = crops[0].gate, p = probs[0];
    crops.slice(1).forEach(c => { if (c.roiIn !== c.roi) c.roiIn.delete(); c.roi.delete(); });
    const roiIn = crops[0].roiIn;

    const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 5);
    const conf = p[order[0]], margin = p[order[0]] - p[order[1]];
    let status, message;
    if (!box && conf < this.cfg.lowConf) {
      status = "no_vehicle";
      message = "No vehicle detected in this image. Try a clearer photo of a car.";
    } else if (conf < this.cfg.lowConf) {
      status = "low_confidence";
      message = `Low confidence (${Math.round(conf * 100)}%). The car may be a model outside the 196 Stanford Cars classes, heavily occluded, or shot from an unusual angle — see the top-5 alternatives.`;
    } else {
      status = "ok";
      message = box ? "Vehicle identified." : "Detector found no vehicle box; classified the full frame.";
    }
    t = now();
    const stages = withStages && status !== "no_vehicle" ? this.stages(roi) : [];
    const stagesMs = now() - t;
    const image = Pipeline.toImageData(cv, rgb);
    const result = {
      status, message,
      top5: order.map(i => ({ class_id: i, ...this.cfg.classes[i], confidence: +(p[i] * 100).toFixed(2) })),
      confidence: +(conf * 100).toFixed(2), margin: +(margin * 100).toFixed(2),
      probs: p, vehicles,
      detection: { box, conf: det.primary ? det.primary.conf : 0, label: det.primary ? det.primary.label : "",
                   all: det.boxes, image_size: [rgb.cols, rgb.rows] },
      illumination: { mean_luma: +illum.mean_luma.toFixed(1), contrast: +illum.contrast.toFixed(1), clahe_applied: gate },
      timings: { detection_ms: det.ms, classical_ms: classicalMs, classifier_ms: classifierMs,
                 stages_ms: stagesMs, total_ms: now() - t0 },
      stages, image,
    };
    result.prediction = result.top5[0];
    if (roiIn !== roi) roiIn.delete();
    roi.delete(); rgb.delete();
    return result;
  }
}

/* ------------------------------------------------------------------ utilities */
function now() { return (typeof performance !== "undefined" ? performance : Date).now(); }
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function iou(a, b) {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = iw * ih;
  const u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0;
}
/** Python's round(): half to even. */
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
function softmax(z, T) {
  let m = -Infinity;
  for (const v of z) m = Math.max(m, v / T);
  const e = new Float64Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] / T - m); s += e[i]; }
  for (let i = 0; i < z.length; i++) e[i] /= s;
  return e;
}
/** numpy.median over uint8 data via histogram. */
function median(data) {
  const h = new Uint32Array(256);
  for (const v of data) h[v]++;
  const n = data.length;
  const kth = k => { let c = 0; for (let i = 0; i < 256; i++) { c += h[i]; if (c > k) return i; } return 255; };
  return n % 2 ? kth((n - 1) / 2) : (kth(n / 2 - 1) + kth(n / 2)) / 2;
}
