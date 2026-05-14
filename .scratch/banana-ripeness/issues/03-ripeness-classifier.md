---
Status: done
---

# 03 — RipenessClassifier — MobileNetV2 backbone + predict() interface

## What to build

Implement a `RipenessClassifier` that wraps a pretrained MobileNetV2 backbone fine-tuned on the 4-class ripeness scheme. It loads weights from a checkpoint at construction time and exposes a single `predict()` call that hides all model internals.

## Acceptance criteria

- [x] Loads a MobileNetV2 backbone and replaces the classification head for 4 classes
- [x] Loads weights from a checkpoint file at construction time
- [x] Interface: `predict(image) → (ripeness_stage, confidence_score)`
- [x] `ripeness_stage` is always one of: `unripe`, `nearly-ripe`, `ripe`, `overripe`
- [x] `confidence_score` is the softmax probability of the top predicted class, in [0, 1]
- [x] Backbone is swappable via config without changing the rest of the pipeline
- [x] Unit tests pass: predict() returns a valid ripeness stage string and confidence score in [0, 1]; output stage is one of the four canonical classes; handles a batch of images

## Blocked by

- `02-image-preprocessor.md`
