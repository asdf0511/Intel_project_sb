"""
SB-Ship Welding Defect Classifier — 1st Model (Combo-A+)
Standalone inference entry point.

Usage examples:
  # Single image
  python infer.py --image /path/to/sample.jpg

  # Whole directory (recursive optional via --recursive)
  python infer.py --dir /path/to/images --recursive --out results.csv

  # Use a different decision threshold
  python infer.py --dir /path/to/images --threshold 0.3937

  # Force CPU
  python infer.py --image sample.jpg --device cpu

  # Use the ONNX variant instead of PyTorch (requires onnxruntime)
  python infer.py --dir imgs --backend onnx

Output columns (CSV / stdout):
  file  pred_4class  p_303_10  p_303_20  p_304_10  p_304_20
        p_defect     decision(PASS|FAIL|REVIEW)

Temperature scaling (T=4.940492) is applied ALWAYS unless --no-temp is set.
Preprocessing is the exact same Resize(224) + ToTensor used at training time
(NO ImageNet Normalize — see model_card.json).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_CARD = json.loads((HERE / "model_card.json").read_text())
CLASS_NAMES = MODEL_CARD["class_names"]               # ['303_10','303_20','304_10','304_20']
DEFECT_IDX  = [CLASS_NAMES.index("303_10"), CLASS_NAMES.index("304_10")]
DEFAULT_T   = float(MODEL_CARD["temperature_scaling"]["T"])
DEFAULT_PTH = HERE / MODEL_CARD["files"]["pytorch_weights"]
DEFAULT_ONNX = HERE / MODEL_CARD["files"]["onnx_weights"]

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Preprocessing — must match training (NO Normalize)
# ---------------------------------------------------------------------------
_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _tfm(img)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class PTBackend:
    def __init__(self, weights_path: str, device: str):
        self.device = torch.device(device)
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device)).cpu()


class ONNXBackend:
    def __init__(self, onnx_path: str, device: str):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit(
                "onnxruntime not installed. pip install onnxruntime-gpu  (or) onnxruntime"
            ) from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
                    else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.sess.run(None, {self.input_name: batch.numpy()})[0]
        return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------
def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / max(T, 1e-8)


def logits_to_probs(logits: torch.Tensor, T: float) -> torch.Tensor:
    return torch.softmax(apply_temperature(logits, T), dim=1)


def decide(p_defect: float, thr: float, review_low: float, review_high: float) -> str:
    if review_low is not None and review_high is not None and review_low < p_defect < review_high:
        return "REVIEW"
    return "FAIL" if p_defect >= thr else "PASS"


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_files(image: str, dir_: str, recursive: bool) -> list:
    if image:
        return [image]
    base = Path(dir_)
    if recursive:
        return sorted(str(p) for p in base.rglob("*") if p.suffix.lower() in _IMG_EXT)
    return sorted(str(p) for p in base.iterdir() if p.suffix.lower() in _IMG_EXT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SB-Ship welding defect classifier (Combo-A+) inference")
    ap.add_argument("--image", help="Single image path")
    ap.add_argument("--dir",   help="Directory of images")
    ap.add_argument("--recursive", action="store_true", help="Recurse into --dir")
    ap.add_argument("--out",   help="Optional CSV output path")
    ap.add_argument("--backend", choices=["pt", "onnx"], default="pt")
    ap.add_argument("--weights", default=None, help="Override .pth/.onnx path")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P_defect ≥ threshold → FAIL. See model_card.json for suggested values.")
    ap.add_argument("--review-low",  type=float, default=0.45)
    ap.add_argument("--review-high", type=float, default=0.60)
    ap.add_argument("--no-review",   action="store_true", help="Disable review-queue routing")
    ap.add_argument("--no-temp",     action="store_true", help="Disable temperature scaling")
    ap.add_argument("-T", "--temperature", type=float, default=DEFAULT_T)
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("provide --image or --dir")

    files = collect_files(args.image, args.dir, args.recursive)
    if not files:
        print("no images found", file=sys.stderr); sys.exit(1)

    if args.backend == "pt":
        weights = args.weights or str(DEFAULT_PTH)
        backend = PTBackend(weights, args.device)
    else:
        weights = args.weights or str(DEFAULT_ONNX)
        backend = ONNXBackend(weights, args.device)

    T = 1.0 if args.no_temp else args.temperature
    review_low  = None if args.no_review else args.review_low
    review_high = None if args.no_review else args.review_high

    header = ["file", "pred_4class",
              "p_303_10", "p_303_20", "p_304_10", "p_304_20",
              "p_defect", "decision"]
    csv_out = None
    csv_writer = None
    if args.out:
        csv_out = open(args.out, "w", newline="")
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(header)
    else:
        print("\t".join(header))

    n = len(files)
    for start in range(0, n, args.batch_size):
        chunk = files[start:start + args.batch_size]
        imgs = torch.stack([load_image(f) for f in chunk])
        logits = backend.forward(imgs)
        probs  = logits_to_probs(logits, T)  # [B, 4]
        preds  = probs.argmax(dim=1).tolist()
        p_def  = (probs[:, DEFECT_IDX[0]] + probs[:, DEFECT_IDX[1]]).tolist()
        for f, pred, p, pd in zip(chunk, preds, probs.tolist(), p_def):
            decision = decide(pd, args.threshold, review_low, review_high)
            row = [f, CLASS_NAMES[pred],
                   f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{p[3]:.4f}",
                   f"{pd:.4f}", decision]
            if csv_writer:
                csv_writer.writerow(row)
            else:
                print("\t".join(row))

    if csv_out:
        csv_out.close()
        print(f"wrote {n} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
