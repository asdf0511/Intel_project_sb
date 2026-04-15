"""
SB-Ship Welding Defect Classifier — FastAPI Server
ai-army 프론트엔드와 연결하기 위한 /inspect 엔드포인트 제공.

Usage:
  cd 1st_Model

  # HTTP (데스크탑)
  .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

  # HTTPS (아이폰 등 모바일 — 인증서 자동 생성)
  python server.py --https

  # 접속 주소 (같은 WiFi 필요)
  http://192.168.x.x:8000     ← 데스크탑
  https://192.168.x.x:8443    ← 아이폰 (경고창에서 '계속 진행' 선택)
"""
import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_CARD = json.loads((HERE / "model_card.json").read_text())
CLASS_NAMES = MODEL_CARD["class_names"]          # ['303_10','303_20','304_10','304_20']
DEFECT_IDX  = [CLASS_NAMES.index("303_10"), CLASS_NAMES.index("304_10")]
DEFAULT_T   = float(MODEL_CARD["temperature_scaling"]["T"])
THRESHOLD   = 0.5

DEFECT_KO = {
    "303_10": "언더컷",
    "304_10": "오버랩",
}

GRADE_THRESHOLDS = [
    (0.2, "A"),
    (0.4, "B"),
    (0.7, "C"),
    (1.0, "D"),
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_model = models.resnet50(weights=None)
_model.fc = nn.Linear(_model.fc.in_features, len(CLASS_NAMES))
_state = torch.load(
    HERE / MODEL_CARD["files"]["pytorch_weights"], map_location="cpu"
)
_model.load_state_dict(_state)
_model.eval().to(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SB-Ship Welding Defect Classifier", version=MODEL_CARD["version"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score_to_grade(p_defect: float) -> str:
    for thr, grade in GRADE_THRESHOLDS:
        if p_defect <= thr:
            return grade
    return "D"


@torch.inference_mode()
def _run(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _tfm(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    logits = _model(tensor)
    inference_ms = int((time.perf_counter() - t0) * 1000)

    probs = torch.softmax(logits / DEFAULT_T, dim=1)[0].cpu().tolist()
    p_defect = probs[DEFECT_IDX[0]] + probs[DEFECT_IDX[1]]
    return probs, p_defect, inference_ms


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/inspect")
async def inspect(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 이미지입니다.")

    try:
        probs, p_defect, inference_ms = _run(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {e}")

    grade = _score_to_grade(p_defect)

    # 불량 클래스만 detections에 포함 (신뢰도 0.05 이상)
    detections = []
    for i, idx in enumerate(DEFECT_IDX):
        conf = probs[idx]
        if conf >= 0.05:
            detections.append({
                "class_id": i,
                "class_name": DEFECT_KO[CLASS_NAMES[idx]],
                "part_name": "용접부",
                "confidence": round(conf, 4),
                "bbox": None,
                "polygon": None,
            })
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "inspection_id": str(uuid.uuid4()),
        "model_id": MODEL_CARD["name"],
        "model_version": MODEL_CARD["version"],
        "overall_grade": grade,
        "severity_score": round(p_defect, 4),
        "inference_ms": inference_ms,
        "detections": detections,
        "image_url": None,
        "overlay_url": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "model_id": MODEL_CARD["name"],
        "device": str(device),
    }


@app.get("/models")
async def list_models():
    return [{
        "model_id": MODEL_CARD["name"],
        "version": MODEL_CARD["version"],
        "task": "classification",
        "is_default": True,
    }]


# ---------------------------------------------------------------------------
# Static files (ai-army 프론트엔드)
# ---------------------------------------------------------------------------
FRONTEND_DIR = HERE.parent / "Downloads" / "ai-army" / "public"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# HTTPS 실행 (python server.py --https)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import subprocess
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--https", action="store_true", help="HTTPS 모드로 실행 (아이폰용)")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    if args.https:
        port = args.port or 8443
        cert = HERE / "cert.pem"
        key  = HERE / "key.pem"
        if not cert.exists():
            print("인증서 생성 중...")
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(cert),
                "-days", "365", "-nodes",
                "-subj", "/CN=weldinspect"
            ], check=True)
            print(f"인증서 생성 완료: {cert}")
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"\n아이폰에서 접속: https://{local_ip}:{port}")
        print("⚠️  경고창이 뜨면 '고급' → '계속 진행' 선택\n")
        uvicorn.run("server:app", host="0.0.0.0", port=port, ssl_keyfile=str(key), ssl_certfile=str(cert))
    else:
        port = args.port or 8000
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"\n데스크탑: http://localhost:{port}")
        print(f"같은 WiFi: http://{local_ip}:{port}\n")
        uvicorn.run("server:app", host="0.0.0.0", port=port)
