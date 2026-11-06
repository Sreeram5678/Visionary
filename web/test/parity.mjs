/*
 * Parity test: run the browser pipeline (web/pipeline.js) in Node on test
 * images and dump predictions, so scripts/check_web_parity.py can compare them
 * with the Python engine and ground truth.
 *
 * Usage (WEBDEPS = folder with node_modules containing onnxruntime-web,
 * @techstark/opencv-js, jpeg-js):
 *   WEBDEPS=/path/to/webdeps node web/test/parity.mjs 300
 */
import { createRequire } from "module";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Pipeline } from "../pipeline.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const req = createRequire(path.join(process.env.WEBDEPS, "package.json"));
const ort = req("onnxruntime-web");
const jpeg = req("jpeg-js");
let cv = req("@techstark/opencv-js");
if (cv instanceof Promise) cv = await cv;
else if (!cv.Mat) await new Promise(r => (cv.onRuntimeInitialized = r));

ort.env.wasm.numThreads = 1;
const N = +(process.argv[2] || 300);
const classes = JSON.parse(fs.readFileSync(path.join(root, "web/data/classes.json")));
const reg = JSON.parse(fs.readFileSync(path.join(root, "web/models/models.json")));
const spec = reg.models[reg.default];
const pipe = new Pipeline(ort, cv, { classes, temperature: spec.temperature, imgSize: spec.imgSize });
await pipe.load(fs.readFileSync(path.join(root, "web/models/yolov8n.onnx")),
                fs.readFileSync(path.join(root, "web", spec.file)));

const lines = fs.readFileSync(path.join(root, "data/prepared/test.csv"), "utf8").trim().split("\n").slice(1);
// deterministic shuffle (LCG) so the same subset is used every run
let s = 12345;
const idx = lines.map((_, i) => i).sort(() => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5).slice(0, N);
const out = [];
let t0 = Date.now();
for (const i of idx) {
  const rel = lines[i].split(",")[1];
  const raw = jpeg.decode(fs.readFileSync(path.join(root, "data/prepared", rel)), { useTArray: true, formatAsRGBA: true });
  const r = await pipe.analyze({ data: new Uint8ClampedArray(raw.data), width: raw.width, height: raw.height }, { withStages: out.length < 3 });
  out.push({ row: i, pred: r.prediction.class_id, conf: r.confidence, box: r.detection.box, gate: r.illumination.clahe_applied,
             top5: r.top5.map(x => x.class_id), ms: r.timings });
}
fs.writeFileSync(path.join(root, "data/web_parity.json"), JSON.stringify(out));
console.log(`done ${out.length} images in ${((Date.now() - t0) / 1000).toFixed(0)}s; avg total ${(
  out.reduce((a, b) => a + b.ms.total_ms, 0) / out.length).toFixed(0)} ms`);
