# Project Context: AI Detection of Banana Ripeness

## Domain Overview

This project uses computer vision and machine learning to automatically detect and classify the ripeness stage of bananas from images. It targets agricultural applications where rapid, accurate, non-destructive ripeness assessment is needed.

## Glossary

| Term | Definition | Avoid |
|---|---|---|
| **Ripeness stage** | The classified maturity level of a banana (e.g. Unripe, Nearly Ripe, Ripe, Overripe) | "quality", "grade" |
| **Unripe** | Green banana, starch not yet converted to sugar | "raw", "green stage" |
| **Nearly Ripe** | Yellow-green banana, beginning to ripen | "turning", "semi-ripe" |
| **Ripe** | Fully yellow banana, optimal for consumption | "mature", "ready" |
| **Overripe** | Brown-spotted or fully brown banana, past peak | "bad", "spoiled" (unless truly spoiled) |
| **ROI** | Region of Interest — the bounding box or mask isolating the banana in an image | "crop", "patch" |
| **Inference** | Running a trained model on a new image to predict ripeness stage | "prediction run", "classification call" |
| **Dataset** | Labeled collection of banana images used for training and evaluation | "training data" (only if specifically the training split) |
| **Confidence score** | The model's probability estimate for its top predicted class | "accuracy", "certainty" |

## Ripeness Stage Classification

The canonical 4-class scheme:

1. `unripe` — green
2. `nearly-ripe` — yellow-green
3. `ripe` — fully yellow
4. `overripe` — brown spots or fully brown

## Dataset

- **Source:** `luischuquimarca/Banana_Ripeness` on Hugging Face
- 3,495 real Cavendish banana images + 161,280 synthetic images
- Loaded via HuggingFace `datasets` library
- Reference paper: VISAPP 2023 (Chuquimarca, Vintimilla, Velastin)

## Mobile Deployment

- **Android:** TFLite model (int8 quantized, ≤ 1MB) via Kotlin + ML Kit
- **iOS:** Core ML model (int8 quantized) via Swift
- Fully offline — model bundled in app, no network calls at inference time
- Reference model: `jorgealbert/modelo-mobilenetv2-clasificador-bananas` (HF)

## Key Constraints

- Images may be taken under varied lighting conditions (field, warehouse, market)
- Model must run on a mid-range mobile phone with no internet connection
- Inference latency target: < 500ms per image on device
- Dataset must be balanced across ripeness stages
- Exported model size: ≤ 1MB (int8 quantized TFLite)
