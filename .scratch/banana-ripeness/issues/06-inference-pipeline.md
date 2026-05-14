---
Status: done
---

# 06 — InferencePipeline — load checkpoint, preprocess, predict under 500ms

## What to build

Implement a lightweight `InferencePipeline` for edge/mobile deployment. It loads a saved checkpoint, accepts a raw image, runs it through the ImagePreprocessor and RipenessClassifier, and returns the ripeness stage and confidence score — all within 500ms.

This is the demoable end-to-end slice: a real banana image goes in, a ripeness stage and confidence score come out.

## Acceptance criteria

- [x] Loads a checkpoint and is ready to predict with minimal setup
- [x] Accepts a raw image as input
- [x] Returns `(ripeness_stage, confidence_score)` for each image
- [x] Inference completes in under 500ms per image on target edge hardware
- [x] Low-confidence predictions are flagged (confidence score surfaced to caller for thresholding)
- [x] Uses ImagePreprocessor — no duplicated preprocessing logic

## Blocked by

- `02-image-preprocessor.md`
- `03-ripeness-classifier.md`
