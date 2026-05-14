---
Status: ready-for-agent
---

# 06 — InferencePipeline — load checkpoint, preprocess, predict under 500ms

## What to build

Implement a lightweight `InferencePipeline` for edge/mobile deployment. It loads a saved checkpoint, accepts a raw image, runs it through the ImagePreprocessor and RipenessClassifier, and returns the ripeness stage and confidence score — all within 500ms.

This is the demoable end-to-end slice: a real banana image goes in, a ripeness stage and confidence score come out.

## Acceptance criteria

- [ ] Loads a checkpoint and is ready to predict with minimal setup
- [ ] Accepts a raw image as input
- [ ] Returns `(ripeness_stage, confidence_score)` for each image
- [ ] Inference completes in under 500ms per image on target edge hardware
- [ ] Low-confidence predictions are flagged (confidence score surfaced to caller for thresholding)
- [ ] Uses ImagePreprocessor — no duplicated preprocessing logic

## Blocked by

- `02-image-preprocessor.md`
- `03-ripeness-classifier.md`
