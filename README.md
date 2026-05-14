# Banana Ripeness AI — Field Detection for Farmers

An end-to-end AI system that classifies banana ripeness from a phone camera photo — fully offline, under 500ms, deployable to any mid-range Android device.

Built for banana farmers in the field: point, tap, get an instant answer.

---

## Repository layout

```
banana-ripeness/
├── src/banana_ripeness/        # Python ML pipeline (train → export)
│   ├── dataset_loader.py       # Load luischuquimarca/Banana_Ripeness from HuggingFace
│   ├── image_preprocessor.py  # Shared resize / normalise / ROI (no train-serve skew)
│   ├── ripeness_classifier.py # MobileNetV2 fine-tuned on 4-class ripeness scheme
│   ├── evaluator.py           # Per-class precision / recall / F1
│   ├── training_pipeline.py   # Train loop → best checkpoint by val F1
│   ├── inference_pipeline.py  # Lightweight predict() with low-confidence flag
│   └── model_exporter.py      # Export to ONNX int8 for mobile deployment
│
├── tests/                      # 87 pytest tests, all offline (no real images)
│
├── mobile/android/             # Android app (Kotlin + CameraX + ONNX Runtime)
│   └── app/src/main/
│       ├── java/com/bananaripeness/
│       │   ├── RipenessClassifier.kt   # On-device ONNX inference
│       │   └── MainActivity.kt         # Camera UI → result card
│       └── res/                        # Layouts, colours, strings
│
├── docs/
│   ├── technical/TECHNICAL.md  # Full technical reference
│   ├── adr/                    # Architectural decision records
│   └── agents/                 # Agent skill configuration
│
├── .scratch/banana-ripeness/   # PRD and implementation issues
├── CONTEXT.md                  # Domain glossary and constraints
└── pyproject.toml
```

---

## Ripeness stages

| Stage | Colour | Description |
|---|---|---|
| `unripe` | Green | Starch not yet converted — not ready |
| `nearly-ripe` | Yellow-green | Beginning to ripen — a few more days |
| `ripe` | Yellow | Peak sweetness — ready to sell |
| `overripe` | Brown | Past peak — sell immediately |

---

## Quickstart

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Train

```python
from banana_ripeness.training_pipeline import TrainingPipeline, TrainingConfig

pipeline = TrainingPipeline(TrainingConfig(epochs=20))
history = pipeline.run()   # downloads dataset from HuggingFace, saves checkpoints/best.pth
```

### 3. Run inference

```python
from PIL import Image
from banana_ripeness.inference_pipeline import InferencePipeline

pipeline = InferencePipeline("checkpoints/best.pth")
result = pipeline.predict(Image.open("banana.jpg"))

print(result.stage)           # "ripe"
print(result.confidence)      # 0.94
print(result.low_confidence)  # False
```

### 4. Export for mobile

```python
from banana_ripeness.model_exporter import ModelExporter

ModelExporter().export(
    "checkpoints/best.pth",
    format="onnx",
    output_dir="mobile/android/app/src/main/assets",
)
```

### 5. Build the Android app

Open `mobile/android/` in Android Studio → sync Gradle → Run on device.

See `mobile/README.md` for full instructions.

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Dataset

**[luischuquimarca/Banana_Ripeness](https://huggingface.co/datasets/luischuquimarca/Banana_Ripeness)**
- 3,495 real Cavendish banana images
- 161,280 synthetic augmented images
- Peer-reviewed: VISAPP 2023 (Chuquimarca, Vintimilla, Velastin)

---

## Model

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (fine-tuned) |
| Parameters | 2.3M |
| Input | 224 × 224 RGB |
| Output | 4-class softmax (ripeness stage + confidence) |
| Export format | ONNX int8 quantized (~2.3MB) |
| Inference runtime | ONNX Runtime Mobile (Android + iOS) |
| Latency target | < 500ms on mid-range device |

---

## Technical documentation

See **[docs/technical/TECHNICAL.md](docs/technical/TECHNICAL.md)** for:
- Full architecture overview
- Module interfaces
- Training and evaluation guide
- Mobile deployment guide
- Architectural decisions (ADRs)

---

## Licence

See [LICENSE](LICENSE).
