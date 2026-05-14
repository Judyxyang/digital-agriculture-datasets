---
Status: ready-for-agent
---

# PRD: AI Detection of Banana Ripeness

## Problem Statement

Banana farmers and agricultural workers currently assess ripeness stage by eye — a process that is slow, inconsistent across individuals, and impractical at scale. There is no reliable, automated way to classify whether a banana is `unripe`, `nearly-ripe`, `ripe`, or `overripe` from a phone camera image, especially under the varied lighting conditions of field, warehouse, and market environments.

## Solution

A mobile app for banana farmers that uses the phone camera to photograph a banana and instantly displays its ripeness stage (`unripe`, `nearly-ripe`, `ripe`, or `overripe`) along with a confidence score — no internet connection required. The app runs a quantized MobileNetV2 model on-device, trained on the `luischuquimarca/Banana_Ripeness` open-source dataset (3,495 real + 161,280 synthetic Cavendish banana images). Inference completes in under 500ms on a mid-range Android or iOS device.

## Dataset

**Source:** [`luischuquimarca/Banana_Ripeness`](https://hf.co/datasets/luischuquimarca/Banana_Ripeness) on Hugging Face

- 3,495 real Cavendish banana images labeled by ripeness stage
- 161,280 synthetic augmented images for generalisation across lighting conditions
- Peer-reviewed (VISAPP 2023 conference paper)
- Loaded via HuggingFace `datasets` library

## User Stories

1. As a banana farmer, I want to open an app, point my camera at a banana, and instantly see its ripeness stage, so that I can make harvest and sorting decisions without expert help.
2. As a banana farmer, I want the app to work offline in the field, so that I am not dependent on mobile data coverage.
3. As a warehouse operator, I want to photograph a banana and receive its ripeness stage instantly, so that I can sort batches without manual inspection.
4. As a field agricultural worker, I want the system to work under natural outdoor lighting, so that I can assess ripeness on the farm without controlled conditions.
5. As a retailer, I want to detect overripe bananas automatically, so that I can remove them from display before they affect customer perception.
6. As a dataset curator, I want to load the `luischuquimarca/Banana_Ripeness` dataset and get balanced train/val/test splits, so that model training is reproducible.
7. As a dataset curator, I want augmentation applied automatically during training (brightness/contrast jitter), so that the model generalises across lighting conditions.
8. As an ML engineer, I want to train a ripeness classifier on the open-source dataset, so that I can produce a model checkpoint ready for mobile deployment.
9. As an ML engineer, I want to evaluate per-class precision, recall, and F1 across all four ripeness stages, so that I can identify which stages the model confuses before deploying.
10. As an ML engineer, I want the model to use MobileNetV2 as its backbone, so that inference stays within the 500ms latency budget on a mid-range phone.
11. As an ML engineer, I want to export the trained model to TFLite (Android) and Core ML (iOS) with int8 quantization, so that the model file is small enough to bundle in a mobile app.
12. As a developer, I want the ImagePreprocessor to handle resizing, normalisation, and ROI extraction consistently between training and inference, so that there is no train/serve skew.
13. As a developer, I want a single `predict(image) → (ripeness_stage, confidence_score)` call, so that I can integrate ripeness detection without understanding model internals.
14. As a quality reviewer, I want confidence scores returned alongside each ripeness stage prediction, so that I can flag low-confidence results for human review.
15. As a system operator, I want inference to complete in under 500ms per image on a mid-range phone, so that the app feels responsive.
16. As a researcher, I want to swap the classifier backbone without changing the rest of the pipeline, so that I can experiment with newer architectures easily.
17. As a farmer using the mobile app, I want the result displayed in plain language with a colour indicator, so that I can understand the result without technical knowledge.

## Implementation Decisions

### Modules

**DatasetLoader**
- Loads the `luischuquimarca/Banana_Ripeness` dataset from Hugging Face using the `datasets` library
- Applies augmentation during training (random brightness/contrast jitter, horizontal flip)
- Returns balanced train / val / test splits as iterable DataLoaders
- Interface: `load(split) → DataLoader`

**ImagePreprocessor**
- Shared preprocessing used by both training and inference paths (resize to 224×224, normalise to ImageNet mean/std, ROI extraction)
- Ensures no train/serve skew — single source of truth for all image transformations
- Interface: `preprocess(raw_image) → tensor`

**RipenessClassifier**
- Wraps a pretrained MobileNetV2 backbone fine-tuned on the 4-class ripeness scheme
- Loads weights from a checkpoint file at construction time
- Interface: `predict(image) → (ripeness_stage, confidence_score)`
- `ripeness_stage` is one of: `unripe`, `nearly-ripe`, `ripe`, `overripe`
- `confidence_score` is the softmax probability for the top predicted class

**TrainingPipeline**
- Orchestrates: DatasetLoader → ImagePreprocessor → RipenessClassifier training loop → checkpoint saving
- Used offline only; not deployed to mobile
- Saves best checkpoint based on validation F1

**InferencePipeline**
- Lightweight runtime path: loads checkpoint → ImagePreprocessor → RipenessClassifier.predict
- Accepts a raw image, returns ripeness stage + confidence score within 500ms

**Evaluator**
- Computes per-class precision, recall, and F1 for all four ripeness stages
- Interface: `evaluate(predictions, labels) → MetricsReport`

**ModelExporter**
- Converts the trained PyTorch checkpoint to TFLite (Android) and Core ML (iOS) formats
- Applies int8 quantization to reduce model size to ~600KB and improve mobile inference speed
- Interface: `export(checkpoint_path, format) → exported_model_path`
- Validates that exported model output matches original PyTorch model on a sample batch

**MobileApp**
- Android (Kotlin) and/or iOS (Swift) camera app that loads the on-device TFLite / Core ML model
- Farmer-facing UI: camera viewfinder → capture button → ripeness stage label + colour indicator + confidence score
- Fully offline — no network calls at inference time
- Colour indicators: green (`unripe`), yellow-green (`nearly-ripe`), yellow (`ripe`), brown (`overripe`)

### Architectural Decisions

- Dataset loaded directly from Hugging Face Hub via `datasets` library — no manual download required
- MobileNetV2 chosen as backbone: 2.3M parameters, compatible with TFLite and Core ML, meets 500ms latency on mid-range phones
- int8 quantization applied at export — reduces model from ~9MB to ~600KB, fits comfortably in a mobile app bundle
- ImagePreprocessor is shared between training and mobile inference preprocessing to eliminate train/serve skew
- Mobile app runs fully offline — model bundled in the app, no API calls required

## Testing Decisions

### What makes a good test

Test external behaviour only — what goes in and what comes out. Do not assert on internal state, private methods, or implementation details. A test should remain valid even if the internal implementation is completely rewritten.

### Modules to test

| Module | What to test |
|---|---|
| `DatasetLoader` | Returns correct split sizes; augmentation applied only to train split; output is balanced across classes |
| `ImagePreprocessor` | Output tensor shape and dtype are correct; pixel values are in normalised range; ROI crop dimensions are as expected |
| `RipenessClassifier` | `predict()` returns a valid ripeness stage and confidence score in [0, 1]; output is one of the four canonical classes |
| `Evaluator` | Metrics correct for known predictions vs labels; handles edge cases (all-correct, all-wrong, missing class) |
| `ModelExporter` | Exported model output matches PyTorch model output within tolerance on a sample batch |

### Prior art

No existing tests in this repo yet. Follow standard pytest conventions; use small synthetic images (solid-colour numpy arrays) rather than real image files in unit tests.

## Out of Scope

- Real-time video stream processing (single image per capture)
- Detection of multiple bananas in one image (single ROI per image)
- Disease or defect detection beyond ripeness stage
- Server-side inference API
- Dataset collection or labeling tooling
- Support for fruit other than bananas
- User accounts, history, or cloud sync

## Further Notes

- Dataset source: `luischuquimarca/Banana_Ripeness` (VISAPP 2023) — 3,495 real + 161,280 synthetic Cavendish banana images
- Low-confidence predictions should be visually flagged in the mobile UI (e.g. grey indicator) prompting the farmer to retake the photo
- All ripeness stage strings in code, issues, tests, and UI must match the canonical 4-class vocabulary exactly: `unripe`, `nearly-ripe`, `ripe`, `overripe`
- Existing MobileNetV2 banana classifier (`jorgealbert/modelo-mobilenetv2-clasificador-bananas` on HF) can be used as a reference for fine-tuning approach
