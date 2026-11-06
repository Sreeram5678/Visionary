/* Visionary — browser UI. Inference runs locally via web/pipeline.js. */
import { Pipeline } from "./pipeline.js";

const $ = s => document.querySelector(s);
const chat = $("#chat"), fileIn = $("#file"), drop = $("#drop");
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tick = () => new Promise(r => setTimeout(r, 16));
const T0 = performance.now();
const log = s => console.log(`[Visionary] +${Math.round(performance.now() - T0)}ms ${s}`);
const MAX_BYTES = 15 * 1024 * 1024;
const CACHE = "visionary-v1";

let reg, classes, pipe = null, current = null, busy = false, ready = null;
const loadedClassifiers = {};

/* ------------------------------------------------ cached download w/ progress */
async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) { onProgress(1, 1); return new Uint8Array(await hit.arrayBuffer()); } } catch (e) { cache = null; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not download ${url} (${r.status})`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onProgress(got, total || got);
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  try { if (cache) await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch (e) { /* quota: fine */ }
  return buf;
}

async function downloadAll(items) {
  const totals = items.map(i => i.size || 1), done = items.map(() => 0);
  const T = totals.reduce((a, b) => a + b, 0);
  const show = label => {
    const pct = Math.min(100, (done.reduce((a, b) => a + b, 0) / T) * 100);
    $("#loaderBar").style.width = pct.toFixed(1) + "%";
    $("#loaderText").textContent = `${label} · ${pct.toFixed(0)}% of ${(T / 1048576).toFixed(0)} MB (first visit only, then cached)`;
  };
  log(`download start (${items.length} files)`);
  return Promise.all(items.map((it, k) => fetchCached(it.url, (g, t) => { done[k] = (g / t) * totals[k]; show(it.label); })
    .then(b => { log(`downloaded ${it.label} (${(b.length / 1048576).toFixed(1)} MB)`); return b; })));
}

async function loadOpenCV(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
  await new Promise((res, rej) => { const s = document.createElement("script"); s.src = url; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
  // Emscripten's Module object defines .then(), so `await cv` (or returning cv
  // from an async function) makes the browser resolve a self-referencing
  // thenable forever and freezes the tab. Poll for readiness instead, then
  // strip .then and hand the module back wrapped in an object.
  for (let i = 0; !(window.cv && window.cv.Mat); i++) {
    if (i > 1200) throw new Error("OpenCV.js failed to initialise");
    await new Promise(r => setTimeout(r, 50));
  }
  const cv = window.cv;
  if (typeof cv.then === "function") delete cv.then;
  return { cv };
}

/* --------------------------------------------------------------- bootstrap */
async function init() {
  try {
    [reg, classes] = await Promise.all([fetch("models/models.json").then(r => r.json()), fetch("data/classes.json").then(r => r.json())]);
    let saved = null; try { saved = localStorage.getItem("visionary-model"); } catch (e) {}
    current = reg.models[saved] ? saved : reg.default;
    renderMenu();
    const spec = reg.models[current];
    const A = reg.assets;
    const [cvBytes, wasm, yolo, cls] = await downloadAll([
      { url: "vendor/opencv.js", label: "OpenCV.js", size: A.opencv },
      { url: "vendor/ort-wasm-simd-threaded.wasm", label: "ONNX Runtime", size: A.wasm },
      { url: "models/yolov8n.onnx", label: "YOLOv8 detector", size: A.yolo },
      { url: spec.file, label: spec.name, size: spec.size },
    ]);
    const step = s => { log(s); $("#loaderText").textContent = s; };
    step("Initialising models…");
    await tick();
    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
    try { ort.env.wasm.wasmBinary = wasm.buffer; } catch (e) {}
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    // Run inference in a Web Worker so the page (spinner, scrolling) stays responsive.
    ort.env.wasm.proxy = true;
    step("Starting OpenCV.js…"); await tick();
    const { cv } = await loadOpenCV(cvBytes);
    step(`Starting ONNX Runtime (${ort.env.wasm.numThreads} threads, isolated=${self.crossOriginIsolated})…`); await tick();
    pipe = new Pipeline(ort, cv, { classes, temperature: spec.temperature, imgSize: spec.imgSize });
    window.visionaryPipeline = pipe;  // exposed for debugging / benchmarking in devtools
    // In proxy (worker) mode ORT *transfers* the buffer to the worker, detaching
    // it — so always hand ORT a copy and keep the original for model switching.
    await pipe.load(yolo, cls.slice());
    loadedClassifiers[current] = cls;
    step("Warming up…"); await tick();
    // warm-up so the first real request is fast
    await pipe.analyze({ data: new Uint8ClampedArray(64 * 64 * 4).fill(128), width: 64, height: 64 }, { withStages: false });
    $("#loader")?.classList.add("done");
    log("ready");
  } catch (e) {
    console.error(e);
    $("#loaderText").textContent = "Could not load the models: " + (e.message || e) + ". Please reload the page.";
    throw e;
  }
}
ready = init();

/* -------------------------------------------------------------- model menu */
function renderMenu() {
  const m = reg.models[current];
  $("#modelName").textContent = m ? m.name : "Model";
  $("#menu").innerHTML = Object.entries(reg.models).map(([k, x]) => `<button data-k="${esc(k)}" class="${k === current ? "sel" : ""}">
    ${esc(x.name)}${k === current ? " ✓" : ""}<small>${esc(x.desc || "")}${x.top1 ? ` · ${x.top1}% top-1` : ""} · ${(x.size / 1048576).toFixed(0)} MB</small></button>`).join("");
}
$("#pickerBtn").onclick = e => { e.stopPropagation(); $("#menu").classList.toggle("open"); };
$("#menu").onclick = async e => {
  const b = e.target.closest("button[data-k]"); if (!b || busy) return;
  const k = b.dataset.k; $("#menu").classList.remove("open");
  if (k === current) return;
  busy = true; $("#sendBtn").disabled = true;
  const prev = current;
  $("#modelName").textContent = "Loading " + reg.models[k].name + "…";
  try {
    await ready;
    const spec = reg.models[k];
    const bytes = loadedClassifiers[k] || await fetchCached(spec.file, (g, t) => { $("#modelName").textContent = `Loading ${Math.round(g / t * 100)}%`; });
    loadedClassifiers[k] = bytes;
    await pipe.setClassifier(bytes.slice(), {}, { temperature: spec.temperature, imgSize: spec.imgSize });
    current = k; try { localStorage.setItem("visionary-model", k); } catch (e) {}
  } catch (err) { current = prev; addError("Could not switch model: " + (err.message || err)); }
  finally { renderMenu(); busy = false; $("#sendBtn").disabled = false; }
};
document.addEventListener("click", () => $("#menu").classList.remove("open"));

/* ------------------------------------------------------------------ input */
const SAMPLES = ["samples/ferrari.jpg", "samples/bmw_dark.jpg", "samples/rangerover.jpg"];
let sampleIdx = 0;
async function trySample() {
  if (busy) return;
  const url = SAMPLES[sampleIdx++ % SAMPLES.length];
  try {
    const blob = await (await fetch(url)).blob();
    send(new File([blob], url.split("/").pop(), { type: "image/jpeg" }));
  } catch (e) { addError("Could not load the sample image."); }
}
$("#sampleBtn").onclick = trySample;
window.visionarySample = trySample;
document.querySelectorAll("[data-pick]").forEach(el => (el.onclick = () => !busy && fileIn.click()));
fileIn.onchange = () => { if (fileIn.files[0]) send(fileIn.files[0]); fileIn.value = ""; };
["dragenter", "dragover"].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
document.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) send(f); });
document.addEventListener("paste", e => { const it = [...e.clipboardData.items].find(i => i.type.startsWith("image/")); if (it) send(it.getAsFile()); });
const scrollEnd = () => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });

async function decode(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch (e) { throw new Error("That file isn't a readable image. Please upload a JPG, PNG or WEBP."); }
  if (Math.min(bmp.width, bmp.height) < 32) throw new Error("Image is too small (minimum 32×32 pixels).");
  const s = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(bmp, 0, 0, w, h); bmp.close && bmp.close();
  return g.getImageData(0, 0, w, h);
}

function imgURL(id) {
  const c = document.createElement("canvas"); c.width = id.width; c.height = id.height;
  c.getContext("2d").putImageData(new ImageData(id.data, id.width, id.height), 0, 0);
  return c.toDataURL("image/jpeg", 0.85);
}

async function send(file) {
  if (busy) return;
  $("#hero")?.classList.add("compact");
  if (!file.type.startsWith("image/")) { $("#hero")?.remove(); addError("That file isn't an image. Please upload a JPG, PNG or WEBP."); return; }
  if (file.size > MAX_BYTES) { $("#hero")?.remove(); addError("Image is larger than 15 MB. Please upload a smaller image."); return; }
  busy = true; $("#sendBtn").disabled = true;
  const hero = $("#hero"); if (hero) { if (pipe) hero.remove(); else hero.querySelector(".samples")?.remove(); }
  chat.insertAdjacentHTML("beforeend", `<div class="msg user"><img src="${URL.createObjectURL(file)}" alt="Uploaded image"></div>`);
  const pending = document.createElement("div");
  pending.className = "msg bot";
  pending.innerHTML = `<div class="dot"></div><div class="body"><div class="typing"><span><i></i> <i></i> <i></i></span><span class="phase">Reading image…</span></div></div>`;
  chat.appendChild(pending); scrollEnd();
  const phase = t => { const p = pending.querySelector(".phase"); if (p) p.textContent = t; };
  try {
    const img = await decode(file);
    if (!pipe) { phase("Waiting for models to finish loading…"); await ready; }
    $("#hero")?.remove();
    phase("Locating vehicle, normalising illumination, classifying…"); await tick(); await tick();
    const d = await pipe.analyze(img);
    d.model = reg.models[current];
    pending.remove();
    renderResult(d);
  } catch (e) {
    console.error(e); pending.remove();
    addError(e.message || "Something went wrong. Please try another image.");
  } finally { busy = false; $("#sendBtn").disabled = false; scrollEnd(); }
}

function addError(msg) {
  chat.insertAdjacentHTML("beforeend", `<div class="msg bot"><div class="dot"></div><div class="body"><div class="notice bad">${esc(msg)}</div></div></div>`);
  scrollEnd();
}

/* ----------------------------------------------------------------- render */
function renderResult(d) {
  const el = document.createElement("div"); el.className = "msg bot";
  const p = d.prediction, c = d.confidence, t = d.timings;
  const cls = c >= 75 ? "ok" : c >= 50 ? "warn" : "bad";
  const head = d.status === "no_vehicle" ? `<div class="notice bad">${esc(d.message)}</div>` : `
    ${d.status === "low_confidence" ? `<div class="notice warn">${esc(d.message)}</div>` : ""}
    <div class="title">${esc(p.year)} ${esc(p.make)} ${esc(p.model)}</div>
    <div class="sub">${esc(p.body_style || "")}${p.body_style ? " · " : ""}${d.status === "low_confidence" ? "best guess" : "most likely match"}${(d.vehicles || []).length > 1 ? ` · largest of ${d.vehicles.length} vehicles` : ""}</div>
    <span class="badge ${cls}">● ${c.toFixed(1)}% confidence</span>
    ${d.detection.box ? `<span class="badge">▢ ${esc(d.detection.label)} detected ${(d.detection.conf * 100).toFixed(0)}%</span>` : `<span class="badge warn">No box: full frame</span>`}
    ${d.illumination.clahe_applied ? `<span class="badge">☀ CLAHE applied (dark/flat image)</span>` : ""}
    <span class="badge">⏱ ${Math.round(t.total_ms)} ms on this device</span>`;
  el.innerHTML = `<div class="dot"></div><div class="body">${head}
    <div class="canvaswrap" style="margin-top:10px"><canvas></canvas></div>
    ${d.status !== "no_vehicle" ? `
    ${vehiclesCard(d)}
    <div class="card"><h3>Top-5 predictions${(d.vehicles || []).length > 1 ? " · vehicle #1" : ""}</h3>
      ${d.top5.map((x, i) => `<div class="row ${i ? "alt" : ""}"><span class="n">${esc(x.year)} ${esc(x.make)} ${esc(x.model)}</span><span class="v">${x.confidence.toFixed(1)}%</span><div class="track"><div class="fill" data-w="${Math.max(1, x.confidence)}"></div></div></div>`).join("")}
    </div>
    <details><summary>How Visionary got this answer</summary>
      <div class="card"><h3>Pipeline (ran on your device)</h3><div class="kv">
        <div><small>Detection (YOLOv8n)</small><b>${Math.round(t.detection_ms)} ms</b></div>
        <div><small>Classical CV gate</small><b>${Math.round(t.classical_ms)} ms</b></div>
        <div><small>Classifier (${esc(d.model.name)})</small><b>${Math.round(t.classifier_ms)} ms</b></div>
        <div><small>Brightness / contrast</small><b>${d.illumination.mean_luma} / ${d.illumination.contrast}</b></div>
        <div><small>Top-1 vs Top-2 margin</small><b>${d.margin.toFixed(1)} pts</b></div>
      </div>
      <div class="stages">${d.stages.map(s => `<div class="stage"><img src="${imgURL(s.img)}" alt="${esc(s.name)}"><b>${esc(s.name)}</b><span>${esc(s.desc)}</span></div>`).join("")}</div>
      </div></details>` : ""}
  </div>`;
  chat.appendChild(el);
  drawBoxes(el.querySelector("canvas"), d);
  requestAnimationFrame(() => el.querySelectorAll(".fill").forEach(f => (f.style.width = f.dataset.w + "%")));
}

const VEH_COLORS = ["--accent", "#2f7dd1", "#2f9e6b", "#b5651d", "#8a4fbf", "#c2417a", "#3a8f8f", "#7a7a2f"];
function vehColor(i) {
  const c = VEH_COLORS[i % VEH_COLORS.length];
  return c.startsWith("--") ? (getComputedStyle(document.documentElement).getPropertyValue(c).trim() || "#c96442") : c;
}

function drawBoxes(cv, d) {
  const im = d.image;
  cv.width = im.width; cv.height = im.height;
  const g = cv.getContext("2d");
  g.putImageData(new ImageData(im.data, im.width, im.height), 0, 0);
  const lw = Math.max(2, cv.width / 260);
  const veh = (d.vehicles || []).filter(v => v.box);
  const multi = veh.length > 1;
  // Unclassified detections (tiny / duplicate boxes): thin white outline.
  for (const b of d.detection.all || []) {
    if (veh.some(v => v.box === b.box)) continue;
    const [x1, y1, x2, y2] = b.box;
    g.lineWidth = lw * 0.6; g.strokeStyle = "rgba(255,255,255,.55)";
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (d.status === "no_vehicle") return;
  const fs = Math.max(13, cv.width / (multi ? 60 : 45));
  g.font = `600 ${fs}px system-ui,sans-serif`;
  // Draw in reverse so the primary (#1) label ends up on top.
  for (const v of veh.slice().reverse()) {
    const [x1, y1, x2, y2] = v.box, col = vehColor(v.index - 1);
    g.lineWidth = v.index === 1 ? lw * 1.6 : lw * 1.2; g.strokeStyle = col;
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const p = v.prediction;
    // Full name only on the main car; others get a number badge (names are in the list).
    const label = v.index === 1
      ? `${multi ? "#1 " : ""}${p.year} ${p.make} ${p.model} · ${v.confidence.toFixed(0)}%`
      : `#${v.index}${v.status === "low_confidence" ? " ?" : ""}`;
    const tw = g.measureText(label).width + fs * 0.8, th = fs * 1.6;
    const ly = y1 - th < 0 ? y1 : y1 - th;
    const lx = Math.max(0, Math.min(x1 - lw * 0.8, cv.width - tw));
    g.fillStyle = col; g.fillRect(lx, ly, tw, th);
    g.fillStyle = "#fff"; g.fillText(label, lx + fs * 0.4, ly + th * 0.7);
  }
}

function vehiclesCard(d) {
  const veh = d.vehicles || [];
  if (veh.length < 2) return "";
  return `<div class="card"><h3>${veh.length} vehicles found</h3>
    ${veh.map(v => `<details class="veh"><summary>
        <span class="vnum" style="background:${vehColor(v.index - 1)}">${v.index}</span>
        <span class="vname">${esc(v.prediction.year)} ${esc(v.prediction.make)} ${esc(v.prediction.model)}</span>
        <span class="vconf ${v.status === "ok" ? "" : "low"}">${v.confidence.toFixed(1)}%${v.status === "ok" ? "" : " · unsure"}</span>
      </summary>
      <div class="vtop">${v.top5.slice(1, 5).map(x => `<div><span>${esc(x.year)} ${esc(x.make)} ${esc(x.model)}</span><span>${x.confidence.toFixed(1)}%</span></div>`).join("")}
        <div class="vmeta">${esc(v.label)} detected ${(v.det_conf * 100).toFixed(0)}%${v.clahe_applied ? " · CLAHE applied" : ""}</div></div>
    </details>`).join("")}</div>`;
}

/* -------------------------------------------------------------- benchmarks */
async function openBench() {
  $("#overlay").classList.add("open");
  const sheet = $("#sheet");
  sheet.innerHTML = `<button class="close" id="closeB">Close</button><h2>Benchmarks</h2><p class="sub">Loading…</p>`;
  $("#closeB").onclick = closeBench;
  try {
    const b = await (await fetch("data/eval_results.json")).json();
    const row = r => `<tr>${r.map(v => `<td>${esc(v ?? "–")}</td>`).join("")}</tr>`;
    sheet.innerHTML = `<button class="close" id="closeB">Close</button>
      <h2>Benchmarks</h2><p class="sub">Official Stanford Cars test set: ${esc(b.n_test)} images never seen in training. Server latency is CPU, 2 threads.</p>
      <div class="card"><h3>Classical vs deep learning vs hybrid</h3><div class="tablewrap"><table>
        <tr><th>Approach</th><th>Top-1 %</th><th>Top-5 %</th><th>Latency ms</th><th>Size MB</th></tr>
        ${b.comparison.map(r => row([r.name, r.top1, r.top5, r.latency_ms, r.size_mb])).join("")}</table></div></div>
      ${b.browser ? `<div class="card"><h3>This in-browser app (${esc(b.browser.model || "default model")}, INT8)</h3><div class="kv">
        <div><small>Top-1 (test set)</small><b>${b.browser.top1}%</b></div><div><small>Top-5</small><b>${b.browser.top5}%</b></div>
        <div><small>Agreement with Python</small><b>${b.browser.parity}%</b></div></div></div>` : ""}
      <div class="card"><h3>Ablation: what each component adds</h3><div class="tablewrap"><table>
        <tr><th>Config</th><th>Top-1 %</th><th>Top-5 %</th><th>Δ Top-1</th></tr>
        ${b.ablation.map(r => row([r.name, r.top1, r.top5, r.delta])).join("")}</table></div></div>
      <div class="card"><h3>Detection</h3><div class="kv">
        <div><small>Mean IoU</small><b>${b.detection.mean_iou}</b></div><div><small>mAP@50</small><b>${b.detection.map50}</b></div>
        <div><small>mAP@50:95</small><b>${b.detection.map50_95}</b></div><div><small>Detection rate</small><b>${b.detection.recall}%</b></div></div></div>
      <div class="card"><h3>Robustness (Top-1 %)</h3><div class="tablewrap"><table>
        <tr><th>Condition</th>${b.robustness.methods.map(m => `<th>${esc(m)}</th>`).join("")}</tr>
        ${b.robustness.rows.map(r => row([r.condition, ...r.values])).join("")}</table></div></div>
      <div class="card"><h3>Confidence</h3><div class="kv">
        <div><small>Calibration error (ECE)</small><b>${b.confidence.ece_hybrid}%</b></div>
        <div><small>Accuracy when confident (≥50%)</small><b>${b.confidence.acc_when_confident}%</b></div>
        <div><small>Accuracy when flagged (&lt;50%)</small><b>${b.confidence.acc_when_flagged}%</b></div>
        <div><small>Flag rate</small><b>${b.confidence.low_conf_rate}%</b></div></div></div>`;
    $("#closeB").onclick = closeBench;
  } catch (e) { sheet.innerHTML += `<div class="notice bad">Could not load benchmarks.</div>`; }
}
function closeBench() { $("#overlay").classList.remove("open"); }
$("#benchBtn").onclick = openBench; $("#benchSample").onclick = openBench;
$("#overlay").onclick = e => { if (e.target.id === "overlay") closeBench(); };
document.addEventListener("keydown", e => { if (e.key === "Escape") closeBench(); });
