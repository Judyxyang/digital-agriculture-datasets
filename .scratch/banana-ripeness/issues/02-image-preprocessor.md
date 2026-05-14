---
Status: done
---

# 02 — ImagePreprocessor — shared resize, normalise, ROI extraction

## What to build

Implement a shared `ImagePreprocessor` used by both the training pipeline and inference pipeline. It handles resizing, pixel normalisation, and ROI extraction. Being shared between both paths eliminates train/serve skew — there is a single source of truth for all image transformations.

## Acceptance criteria

- [x] Resizes images to a consistent target resolution
- [x] Normalises pixel values to the expected range
- [x] Extracts the ROI (bounding box isolating the banana) from the full image
- [x] Interface: `preprocess(raw_image) → tensor`
- [x] Unit tests pass: output tensor shape and dtype are correct; pixel values are in normalised range; ROI crop dimensions are as expected
- [x] Used by both TrainingPipeline and InferencePipeline (no duplication)

## Blocked by

None — can start immediately
