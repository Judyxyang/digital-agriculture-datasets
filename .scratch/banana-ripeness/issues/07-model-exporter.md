---
Status: done
---

# 07 — ModelExporter — TFLite & Core ML export with int8 quantization

## What to build

Implement a `ModelExporter` that converts the trained PyTorch MobileNetV2 checkpoint into mobile-ready formats: TFLite (Android) and Core ML (iOS). It applies int8 quantization to shrink the model from ~9MB to ~600KB and validates that the exported model's outputs match the original PyTorch model within an acceptable tolerance.

## Acceptance criteria

- [x] Exports trained PyTorch checkpoint to TFLite format (Android)
- [x] Exports trained PyTorch checkpoint to Core ML format (iOS)
- [x] Applies int8 quantization to both exported formats
- [x] Exported TFLite model is ≤ 1MB
- [x] Interface: `export(checkpoint_path, format) → exported_model_path`
- [x] Validates exported model output matches original PyTorch model on a sample batch (within floating point tolerance)
- [x] Unit tests pass: exported model output matches PyTorch model output on synthetic input

## Blocked by

- `05-training-pipeline.md` (produces the checkpoint to export)
