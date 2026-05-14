# Technical Reference — Banana Ripeness AI

## Table of contents

1. [System overview](#1-system-overview)
2. [Dataset](#2-dataset)
3. [Module reference](#3-module-reference)
4. [Training guide](#4-training-guide)
5. [Evaluation](#5-evaluation)
6. [Inference](#6-inference)
7. [Mobile export and deployment](#7-mobile-export-and-deployment)
8. [Android app architecture](#8-android-app-architecture)
9. [Testing](#9-testing)
10. [Architectural decisions](#10-architectural-decisions)

---

## 1. System overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (offline)                       │
│                                                                 │
│  HuggingFace         ImagePre-      MobileNetV2    Evaluator   │
│  Banana_Ripeness  →  processor   →  fine-tune   →  val F1   →  │
│  (DatasetLoader)     (shared)       (Classifier)   checkpoint  │
└─────────────────────────────────────────────────────────────────┘
                                                        │
                                               ModelExporter
                                               (ONNX int8)
                                                        │
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT (on device)                     │
│                                                                 │
│  Camera image  →  ImagePre-  →  ONNX Runtime  →  Prediction    │
│  (Android)        processor     (on-device)       + UI card    │
└─────────────────────────────────────────────────────────────────┘
```

**Key design principle:** `ImagePreprocessor` is shared between the training and inference paths. Both paths call `preprocess(image, roi_box=None) → tensor` with identical parameters, eliminating train/serve skew.

---

## 2. Dataset

### Source

**[luischuquimarca/Banana_Ripeness](https://huggingface.co/datasets/luischuquimarca/Banana_Ripeness)**

| Split | Images | Notes |
|---|---|---|
| Real | 3,495 | Cavendish bananas, field and market conditions |
| Synthetic | 161,280 | Augmented from real images for lighting generalisation |

Reference: Chuquimarca, Vintimilla, Velastan — VISAPP 2023.

### Class scheme

| Index | Stage | Visual cue |
|---|---|---|
| 0 | `unripe` | Solid green |
| 1 | `nearly-ripe` | Yellow-green |
| 2 | `ripe` | Fully yellow |
| 3 | `overripe` | Brown spots or fully brown |

### Loading

```python
from banana_ripeness.dataset_loader import load

train_loader = load("train", batch_size=32)
val_loader   = load("validation", batch_size=32)
test_loader  = load("test", batch_size=32)
```

- Train split: brightness/contrast jitter (±30%) + random horizontal flip
- Val/test splits: deterministic resize + normalise only
- Warns if any class deviates more than 43% from the expected balanced count

---

## 3. Module reference

### `DatasetLoader` — `src/banana_ripeness/dataset_loader.py`

| Symbol | Type | Description |
|---|---|---|
| `RIPENESS_STAGES` | `tuple[str, ...]` | `("unripe", "nearly-ripe", "ripe", "overripe")` |
| `load(split, batch_size, num_workers)` | `→ DataLoader` | Load HuggingFace split as a DataLoader |

---

### `ImagePreprocessor` — `src/banana_ripeness/image_preprocessor.py`

Shared between training and inference. Constructed once; stateless after `__post_init__`.

| Parameter | Default | Description |
|---|---|---|
| `target_size` | `(224, 224)` | Output (H, W) |
| `mean` | ImageNet | Per-channel normalisation mean |
| `std` | ImageNet | Per-channel normalisation std |

**Method:**

```python
preprocessor.preprocess(image: PIL.Image, roi_box: tuple | None = None) → torch.Tensor
# Returns float32 tensor of shape (3, H, W)
```

`roi_box` is `(left, upper, right, lower)` in pixel coordinates. When provided the image is cropped to the box before resizing, isolating the banana from background clutter.

---

### `RipenessClassifier` — `src/banana_ripeness/ripeness_classifier.py`

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_path` | `None` | Path to a `.pth` state dict. Random weights if `None`. |
| `backbone` | `None` | Custom `nn.Module` to replace MobileNetV2. |
| `device` | auto | `"cpu"` or `"cuda"`. Auto-detected. |
| `pretrained` | `True` | Load ImageNet pretrained MobileNetV2 weights. |

**Method:**

```python
classifier.predict(image: PIL.Image, roi_box=None) → (ripeness_stage: str, confidence: float)
```

`confidence` is the softmax probability of the top class, in [0, 1].

**Backbone swap:**

```python
import torchvision.models as models
backbone = models.efficientnet_b0(weights=None)
backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 4)
clf = RipenessClassifier(backbone=backbone)
```

---

### `Evaluator` — `src/banana_ripeness/evaluator.py`

```python
evaluator = Evaluator()
report = evaluator.evaluate(predictions, labels)
# predictions and labels: list of int (0–3) or str stage names
print(report)
```

**`MetricsReport` fields:**

| Field | Type | Description |
|---|---|---|
| `per_class` | `list[ClassMetrics]` | One entry per ripeness stage |
| `macro_f1` | `float` | Unweighted mean F1 across all 4 classes |

**`ClassMetrics` fields:** `stage`, `precision`, `recall`, `f1`, `support`

---

### `TrainingPipeline` — `src/banana_ripeness/training_pipeline.py`

```python
from banana_ripeness.training_pipeline import TrainingPipeline, TrainingConfig

config = TrainingConfig(
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    seed=42,
    checkpoint_dir="checkpoints",
    pretrained=True,         # use ImageNet pretrained MobileNetV2
)
pipeline = TrainingPipeline(config)
history = pipeline.run()
# → list[EpochResult(epoch, train_loss, val_macro_f1, saved_checkpoint)]
```

- Optimizer: Adam
- LR schedule: CosineAnnealingLR over `epochs`
- Best checkpoint saved to `<checkpoint_dir>/best.pth` by validation macro F1
- Pass `train_loader` and `val_loader` explicitly to skip HuggingFace download

---

### `InferencePipeline` — `src/banana_ripeness/inference_pipeline.py`

```python
from banana_ripeness.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    checkpoint_path="checkpoints/best.pth",
    low_confidence_threshold=0.6,   # flag predictions below this
)
result = pipeline.predict(pil_image, roi_box=None)
```

**`Prediction` fields:**

| Field | Type | Description |
|---|---|---|
| `stage` | `str` | One of `RIPENESS_STAGES` |
| `confidence` | `float` | Softmax probability in [0, 1] |
| `low_confidence` | `bool` | `True` when `confidence < threshold` |

---

### `ModelExporter` — `src/banana_ripeness/model_exporter.py`

```python
from banana_ripeness.model_exporter import ModelExporter

path = ModelExporter().export(
    checkpoint_path="checkpoints/best.pth",
    format="onnx",                  # "onnx" | "tflite"* | "coreml"*
    output_dir="exports",
    validate=True,                  # verify output matches PyTorch within 1e-4
)
# → Path("exports/model_int8.onnx")
```

\* `tflite` requires `pip install tensorflow`; `coreml` requires `pip install coremltools`.

**What the exporter does:**
1. Exports FP32 ONNX via PyTorch dynamo exporter (opset 18)
2. Applies int8 dynamic quantization via ONNX Runtime
3. Validates: runs both PyTorch and ONNX Runtime on the same dummy input, asserts outputs are within `atol=1e-4`
4. Removes the intermediate FP32 file

---

## 4. Training guide

### Full training run

```bash
# Install
pip install -e ".[dev]"

# Train (downloads dataset automatically on first run)
python - <<'EOF'
from banana_ripeness.training_pipeline import TrainingPipeline, TrainingConfig

history = TrainingPipeline(TrainingConfig(
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    seed=42,
    checkpoint_dir="checkpoints",
    pretrained=True,
)).run()

for r in history:
    print(f"Epoch {r.epoch:>2}  loss={r.train_loss:.4f}  val_f1={r.val_macro_f1:.4f}"
          + (f"  ✓ {r.saved_checkpoint}" if r.saved_checkpoint else ""))
EOF
```

### Hyperparameter tuning tips

| Parameter | Suggested range | Notes |
|---|---|---|
| `learning_rate` | `1e-4` – `3e-3` | Lower for pretrained backbone |
| `epochs` | 10 – 30 | Watch val F1 plateau |
| `batch_size` | 16 – 64 | Reduce if GPU OOM |

### Training on GPU

```python
# TrainingPipeline auto-detects CUDA
config = TrainingConfig(epochs=20)
pipeline = TrainingPipeline(config)
pipeline.run()
```

---

## 5. Evaluation

```python
from banana_ripeness.evaluator import Evaluator
from banana_ripeness.dataset_loader import load
from banana_ripeness.ripeness_classifier import RipenessClassifier

classifier = RipenessClassifier("checkpoints/best.pth")
evaluator  = Evaluator()
test_loader = load("test", batch_size=32, num_workers=2)

preds, labels = [], []
for images, batch_labels in test_loader:
    for img_tensor, label in zip(images, batch_labels):
        from torchvision.transforms.functional import to_pil_image
        stage, _ = classifier.predict(to_pil_image(img_tensor))
        from banana_ripeness.ripeness_classifier import RIPENESS_STAGES
        preds.append(RIPENESS_STAGES.index(stage))
        labels.append(label.item())

report = evaluator.evaluate(preds, labels)
print(report)
```

**Example output:**

```
Ripeness Stage    Precision  Recall  F1     Support
----------------------------------------------------
unripe            0.923      0.941   0.932  256
nearly-ripe       0.871      0.849   0.860  248
ripe              0.956      0.963   0.959  261
overripe          0.904      0.911   0.908  235

Macro F1: 0.915
```

---

## 6. Inference

### Python

```python
from PIL import Image
from banana_ripeness.inference_pipeline import InferencePipeline

pipeline = InferencePipeline("checkpoints/best.pth")

img = Image.open("field_photo.jpg")
result = pipeline.predict(img)

print(f"Stage:      {result.stage}")
print(f"Confidence: {result.confidence:.1%}")
if result.low_confidence:
    print("⚠ Low confidence — retake in better lighting")
```

### With ROI

```python
# If you know where the banana is in the image:
result = pipeline.predict(img, roi_box=(120, 80, 450, 320))
```

### Latency

| Device | Measured latency |
|---|---|
| Python CPU (warm) | < 100ms |
| Android mid-range (ONNX Runtime) | < 500ms |

---

## 7. Mobile export and deployment

### Export

```python
from banana_ripeness.model_exporter import ModelExporter

ModelExporter().export(
    "checkpoints/best.pth",
    format="onnx",
    output_dir="mobile/android/app/src/main/assets",
)
# writes: mobile/android/app/src/main/assets/model_int8.onnx
```

### ONNX model properties

| Property | Value |
|---|---|
| Format | ONNX opset 18, int8 dynamic quantization |
| File size | ~2.3MB |
| Input | `image`: float32 `[1, 3, 224, 224]` (ImageNet normalised) |
| Output | `logits`: float32 `[1, 4]` (raw logits, apply softmax) |
| Preprocessing | Must match `ImagePreprocessor` — resize 224×224, ImageNet mean/std |

### Build the Android app

1. Open `mobile/android/` in **Android Studio Hedgehog** (2023.1) or later
2. Wait for Gradle sync to complete
3. Connect an Android device (API 26+) or start an emulator
4. Press **Run ▶**
5. Grant camera permission on first launch

### Runtime dependencies (Android)

| Library | Version |
|---|---|
| ONNX Runtime Android | 1.18.0 |
| CameraX | 1.3.3 |
| Kotlin Coroutines | 1.8.0 |

---

## 8. Android app architecture

```
MainActivity
├── CameraX (PreviewView + ImageCapture)
│   └── captureAndClassify()
│       ├── takePicture() → ImageProxy → Bitmap
│       └── Dispatchers.Default → RipenessClassifier.predict(bitmap)
│           ├── bitmapToTensor()     (resize → ImageNet normalise → FloatBuffer)
│           ├── OrtSession.run()     (ONNX Runtime, model from assets)
│           └── softmax(logits)      → (stage, confidence)
└── showResult(Prediction)
    ├── MaterialCardView (colour = stage colour)
    ├── stageLabel       (plain-language description)
    ├── confidenceLabel  (percentage)
    └── lowConfidenceWarning (shown if confidence < 0.6)
```

### Preprocessing in Kotlin (`RipenessClassifier.kt`)

Mirrors `ImagePreprocessor.preprocess()` exactly:

```kotlin
// resize to 224×224
val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

// extract RGB pixels → normalise with ImageNet mean/std → CHW FloatBuffer
val r[i] = ((px shr 16 and 0xFF) / 255f - 0.485f) / 0.229f
val g[i] = ((px shr  8 and 0xFF) / 255f - 0.456f) / 0.224f
val b[i] = ((px        and 0xFF) / 255f - 0.406f) / 0.225f
// buffer layout: [R plane][G plane][B plane]
```

### Colour scheme

| Stage | Hex | Android resource |
|---|---|---|
| `unripe` | `#2E7D32` | `@color/stage_unripe` |
| `nearly-ripe` | `#9E9D24` | `@color/stage_nearly_ripe` |
| `ripe` | `#F9A825` | `@color/stage_ripe` |
| `overripe` | `#5D4037` | `@color/stage_overripe` |

---

## 9. Testing

### Run all tests

```bash
python -m pytest tests/ -v
```

### Test breakdown

| Test file | Module tested | Tests | Strategy |
|---|---|---|---|
| `test_dataset_loader.py` | `DatasetLoader` | 12 | Mocked HuggingFace splits, synthetic images |
| `test_image_preprocessor.py` | `ImagePreprocessor` | 15 | Solid-colour PIL images, pixel math |
| `test_ripeness_classifier.py` | `RipenessClassifier` | 12 | Random-weight MobileNetV2, no download |
| `test_evaluator.py` | `Evaluator` | 13 | Known label arrays with hand-computed metrics |
| `test_training_pipeline.py` | `TrainingPipeline` | 9 | Synthetic TensorDataset DataLoaders |
| `test_inference_pipeline.py` | `InferencePipeline` | 17 | Random-weight model, latency assertion |
| `test_model_exporter.py` | `ModelExporter` | 11 | Full ONNX round-trip, output tolerance check |
| **Total** | | **89** | All offline — no real images, no network |

### Design principles

- **Test external behaviour only.** No assertions on private methods or internal state.
- **No real images.** Synthetic `numpy` arrays and solid-colour PIL images keep tests fast and dependency-free.
- **No network.** HuggingFace calls are mocked; pretrained weights not downloaded.
- **Latency assertion.** `test_inference_pipeline.py::TestLatency::test_inference_under_500ms` asserts wall-clock inference < 500ms after a warm-up call.

---

## 10. Architectural decisions

See `docs/adr/` for full decision records.

| ADR | Decision |
|---|---|
| [ADR-0001](../adr/0001-mobilenetv2-backbone.md) | MobileNetV2 as backbone |
| [ADR-0002](../adr/0002-shared-preprocessor.md) | Single shared ImagePreprocessor |
| [ADR-0003](../adr/0003-onnx-export-format.md) | ONNX as mobile export format |
| [ADR-0004](../adr/0004-local-issue-tracker.md) | Local markdown issue tracker |
