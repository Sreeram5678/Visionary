"""
Model optimisation: export the fine-tuned classifier to ONNX (FP32) and apply
dynamic INT8 weight quantisation with ONNX Runtime.

Usage:
    python scripts/export_onnx.py --ckpt weights/convnext_tiny_224.pt
Produces weights/<name>_fp32.onnx and weights/<name>_int8.onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification.model import VehicleClassifier  # noqa: E402


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "weights/convnext_tiny_224.pt"))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = VehicleClassifier(ck["backbone"], 196, pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()}, strict=True)
    model.eval()
    s = int(ck.get("img_size", 224))
    fp32 = ckpt.with_name(ckpt.stem + "_fp32.onnx")
    int8 = ckpt.with_name(ckpt.stem + "_int8.onnx")

    dummy = torch.randn(1, 3, s, s)
    torch.onnx.export(LogitsOnly(model), (dummy,), str(fp32), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = ckpt.with_name(ckpt.stem + "_pre.onnx")
    try:
        quant_pre_process(str(fp32), str(pre))
        src = pre
    except Exception as e:  # pre-processing is an optimisation, not required
        print("quant_pre_process skipped:", e)
        src = fp32
    quantize_dynamic(str(src), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    if pre.exists():
        pre.unlink()

    # Sanity check: INT8 logits agree with PyTorch.
    import onnxruntime as ort
    x = np.random.randn(4, 3, s, s).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = {}
    for p in (fp32, int8):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input": x})[0]
        t = time.perf_counter()
        for _ in range(10):
            sess.run(None, {"input": x[:1]})
        out[p.name] = {"MB": round(p.stat().st_size / 2**20, 1),
                       "cpu_ms": round((time.perf_counter() - t) * 100, 1),
                       "argmax_agree": float((y.argmax(1) == ref.argmax(1)).mean()),
                       "max_abs_diff": float(np.abs(y - ref).max())}
    meta = {"img_size": s, "temperature": ck.get("temperature", 1.0), "backbone": ck["backbone"],
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    int8.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
