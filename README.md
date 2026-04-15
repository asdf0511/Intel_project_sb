# SB-Ship Welding Defect Classifier — 1st Model (Combo-A+)

종합평가보고서(`KR_종합평가보고서_20260415.md`) **즉시 배포 권고 모델**.
조합 = **ResNet-50 (4-class) + Temperature Scaling (T=4.940492)**.

Swin/EfficientDet Cascade 조합은 평가 결과 **정확도를 오히려 떨어뜨려** 배포 대상에서 제외되었습니다. 1차 모델은 단일 ResNet-50 + Temperature Scaling이 현재 시점에서 가장 강력한 production baseline입니다.

---

## 디렉토리 구성

```
1st_Model/
├── README.md                ← 본 문서
├── model_card.json          ← 아키텍처/메트릭/벤치마크/권장 임계값
├── calibration_results.json ← Temperature Scaling 학습 결과 (T=4.940492, ECE 등)
├── infer.py                 ← 단독 실행 가능한 inference 스크립트
└── weights/
    ├── Best_ship_model.pth  ← PyTorch state_dict (91 MB)
    └── resnet50_ts.onnx     ← ONNX export (T 미포함, 동일 logits 출력 — 90 MB)
```

---

## 핵심 성능 지표 (val_holdout 3,126장, Temperature Scaling 후)

| 지표 | 값 |
|---|---:|
| 4-class Accuracy | **80.13%** |
| Defect-binary F1 @ 0.5 | **91.97%** |
| Defect-binary PR-AUC | **0.9663** |
| Defect-binary ROC-AUC | **0.9802** |
| ECE (calibration error) | **0.0233** (보정 전 0.1587 → -85%) |
| TP / FP / FN / TN | 1128 / 91 / 106 / 1801 |

### 권장 임계값

| 시나리오 | threshold | Precision | Recall | 비고 |
|---|---|---|---|---|
| **기본** | 0.5 | 92.53% | 91.41% | 운영 기본값 |
| **균형 (Youden's J)** | **0.3937** | 90.20% | **94.00%** | 안전-정밀 균형, 권장 |
| 안전 위주 (Recall ≥95%) | 0.0258 | 87.2% | ~95% | FP가 240건 수준으로 증가 |

### Review Queue (Human-in-the-loop)

- **정책**: `0.45 < P(defect) < 0.60` → 사람 검토 큐로 라우팅
- **큐 비율**: 13.7% (전체 이미지 중)
- **에러 포착률**: **33.82%** (전체 오답의 1/3을 사람 검토로 보낼 수 있음)

### RTX A6000 벤치마크

| 배치 | 지연(ms/batch) | 처리량 | VRAM |
|---|---:|---:|---:|
| 1 | 6.38 | 156.7 img/s | 393 MiB |
| 8 | 6.66 | 1,201.8 img/s | 275 MiB |
| 32 | 21.47 | 1,490.6 img/s | 445 MiB |
| **128** | **78.96** | **1,620.9 img/s** | **1,437 MiB** |
| 256 | 159.86 | 1,601.4 img/s | 2,785 MiB |

> ⚡ 최대 처리량은 **배치 128**에서 약 **1,621 img/s** (≈ 0.62 ms/image).

---

## 사용법

### 0. 환경

```bash
# 사내 서버 기본 환경 그대로 사용
PY=/storage/busan01/miniconda3/envs/sbship/bin/python3
```

필요 패키지: `torch 1.13+`, `torchvision`, `Pillow`. (onnx 백엔드 사용 시 `onnxruntime-gpu` 추가)

### 1. 단일 이미지

```bash
cd /storage/busan01/snap/snapd-desktop-integration/sbship/welding_defect/1st_Model
$PY infer.py --image /path/to/sample.jpg
```

출력 예시:
```
file                 pred_4class  p_303_10  p_303_20  p_304_10  p_304_20  p_defect  decision
/path/to/sample.jpg  303_10       0.7234    0.0812    0.1501    0.0453    0.8735    FAIL
```

### 2. 디렉토리 일괄 추론

```bash
$PY infer.py --dir /path/to/images --recursive --out results.csv --batch-size 128
```

- `--recursive`: 하위 폴더 포함
- `--batch-size 128`: 최대 처리량 설정
- `--out results.csv`: CSV 저장 (없으면 stdout에 TAB 구분 출력)

### 3. 임계값 / 모드 변경

```bash
# Youden's J 균형 임계값 (권장)
$PY infer.py --dir imgs --threshold 0.3937

# Recall ≥95% (안전 우선, FP 증가 감수)
$PY infer.py --dir imgs --threshold 0.0258

# Review queue 비활성화 (PASS/FAIL만)
$PY infer.py --dir imgs --no-review

# Temperature scaling 끄기 (순수 4-class softmax — 학습 후 보정 전 상태)
$PY infer.py --dir imgs --no-temp
```

### 4. ONNX Runtime 백엔드

PyTorch가 없는 환경(컨테이너 경량화, 엣지 디바이스, Windows 등)에서:

```bash
pip install onnxruntime-gpu  # 또는 onnxruntime (CPU)
$PY infer.py --dir imgs --backend onnx
```

> 주의: ONNX 모델은 logits만 출력합니다. infer.py가 동일한 Temperature/Softmax 후처리를 파이썬 쪽에서 적용합니다.

### 5. 프로그램적 사용 (라이브러리로)

```python
import torch, torch.nn as nn, torchvision.models as models
import torchvision.transforms as T
from PIL import Image

CLASSES = ['303_10','303_20','304_10','304_20']
T_VAL = 4.940491676330566

# --- 전처리: 학습과 동일 (NO Normalize) ---
tfm = T.Compose([T.Resize((224,224)), T.ToTensor()])

# --- 모델 로드 ---
m = models.resnet50(weights=None)
m.fc = nn.Linear(m.fc.in_features, 4)
m.load_state_dict(torch.load("1st_Model/weights/Best_ship_model.pth", map_location="cpu"))
m.eval().cuda()

# --- 추론 ---
img = tfm(Image.open("sample.jpg").convert("RGB")).unsqueeze(0).cuda()
with torch.inference_mode():
    logits = m(img)
    probs  = torch.softmax(logits / T_VAL, dim=1).cpu()[0]

p_defect = probs[CLASSES.index('303_10')].item() + probs[CLASSES.index('304_10')].item()
pred     = CLASSES[int(probs.argmax())]
decision = "FAIL" if p_defect >= 0.5 else "PASS"
print(pred, p_defect, decision)
```

---

## 주의사항

1. **Preprocessing**: 반드시 **Resize(224) + ToTensor** 만. ImageNet `Normalize`를 추가하면 학습과 달라져 **예측이 완전히 망가집니다**.
2. **Temperature**: `T=4.940492`를 `logits / T` 형태로 적용한 후 softmax. `infer.py`는 기본값으로 적용하지만, 다른 코드에서 직접 로직 구현 시 잊지 마십시오.
3. **2-class → 4-class 매핑**: 이 모델은 4-class (언더컷/오버랩 × 불량/양품)입니다. `P(defect) = P(303_10) + P(304_10)` 으로 이진 판정합니다.
4. **Cascade 금지**: Swin / EffDet 기반 Cascade는 **정확도 -17~31 pp 저하**가 검증되어 있어 이 모델과 결합하지 않습니다 (`CS_Report_Phase1D/2/3 참조`).

---

## 관련 문서

- `../KR_종합평가보고서_20260415.md` — Combo-A+ 배포 권고 근거
- `../CS_Report_Phase1A_20260415.md` — ResNet-50 단독 평가
- `../CS_Report_Phase1A_Plus_20260415.md` — Temperature Scaling 보정 결과
- `../CS_Report_Phase1D_20260415.md` — Cascade 실패 분석
- `../CS_Report_Phase4_20260415.md` — ONNX export + 벤치마크
