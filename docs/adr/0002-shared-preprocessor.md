# ADR-0002 — Single shared ImagePreprocessor for training and inference

**Date:** 2026-05-14  
**Status:** Accepted

## Context

Image preprocessing (resize, normalise, ROI crop) must be applied identically during training and inference. If the two paths diverge — even in parameter values or operation order — the model receives inputs at inference that it was never trained on, degrading accuracy. This is called train/serve skew.

## Decision

Implement a single `ImagePreprocessor` class used by both `TrainingPipeline` (via `DatasetLoader`) and `InferencePipeline` (and `RipenessClassifier.kt` on Android). The constants `TARGET_SIZE`, `_IMAGENET_MEAN`, and `_IMAGENET_STD` are defined once and imported wherever needed.

The Android Kotlin implementation mirrors the Python implementation parameter-for-parameter:

```
Python:  (pixel / 255 - mean[c]) / std[c]
Kotlin:  ((px shr shift and 0xFF) / 255f - mean[c]) / std[c]
```

## Rationale

The alternative — duplicating preprocessing logic in training code and inference code — creates a maintenance burden and a silent failure mode. Any divergence (e.g. different normalisation constants, different resize interpolation) produces accuracy degradation that is difficult to diagnose.

## Consequences

- Any preprocessing change (e.g. moving to different normalisation statistics) requires updating one place: `image_preprocessor.py` and its Kotlin mirror in `RipenessClassifier.kt`
- Tests for `ImagePreprocessor` verify exact pixel values, catching any silent drift
- Shared constants (`TARGET_SIZE`, `_IMAGENET_MEAN`, `_IMAGENET_STD`) are exported and used in both `dataset_loader.py` and `inference_pipeline.py`
