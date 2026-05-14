# Banana Ripeness — Mobile App

Offline Android app for banana farmers. Point camera at a banana → get ripeness stage + confidence score instantly, no internet required.

## How to build

### 1. Export the model

Run this from the repo root after training:

```bash
python - <<'EOF'
from banana_ripeness.model_exporter import ModelExporter
ModelExporter().export(
    "checkpoints/best.pth",
    format="onnx",
    output_dir="mobile/android/app/src/main/assets",
)
EOF
```

This writes `model_int8.onnx` into the Android assets folder so it is bundled in the APK.

### 2. Open in Android Studio

1. Open `mobile/android/` as a project in Android Studio Hedgehog or later.
2. Wait for Gradle sync.
3. Plug in a device (Android 8.0+) or start an emulator.
4. Press **Run**.

### 3. Grant camera permission

On first launch, tap **Allow** when prompted for camera access.

## App flow

```
Launch → Camera viewfinder
  ↓
Tap capture button (○)
  ↓
[< 500ms inference on device]
  ↓
Result card:
  ┌──────────────────────────────┐
  │  🟡  Ripe — ready to sell   │
  │      Confidence: 94%        │
  └──────────────────────────────┘

  If confidence < 60%:
  ⚠ Low confidence — try again in better lighting
  [📷 Take another photo]
```

## Colour indicators

| Stage | Colour | Meaning |
|---|---|---|
| `unripe` | Green | Not ready yet |
| `nearly-ripe` | Yellow-green | A few more days |
| `ripe` | Yellow | Ready to sell |
| `overripe` | Brown | Sell immediately |

## Model

- Architecture: MobileNetV2 fine-tuned on `luischuquimarca/Banana_Ripeness`
- Format: ONNX int8 quantized (~2.3MB), served via ONNX Runtime for Android
- Inference: fully offline, < 500ms on mid-range devices

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| CameraX | 1.3.3 | Camera viewfinder + capture |
| ONNX Runtime Android | 1.18.0 | On-device inference |
| Material Components | 1.12.0 | UI components |
