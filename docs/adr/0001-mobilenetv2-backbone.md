# ADR-0001 — MobileNetV2 as classifier backbone

**Date:** 2026-05-14  
**Status:** Accepted

## Context

We need a CNN backbone to classify banana ripeness into 4 stages from 224×224 RGB images. The model must run on a mid-range Android phone (< 500ms inference, < 5MB bundled), so large architectures like ResNet-50 or Vision Transformers are unsuitable.

## Decision

Use **MobileNetV2** pre-trained on ImageNet, fine-tuned end-to-end on the banana ripeness dataset. Replace the default 1000-class head with a 4-class linear classifier.

## Rationale

| Option | Params | ONNX int8 size | Notes |
|---|---|---|---|
| MobileNetV2 | 2.3M | ~2.3MB | ✓ Chosen — mobile-native design |
| EfficientNet-B0 | 5.3M | ~5MB | Larger, marginal accuracy gain |
| ResNet-18 | 11.7M | ~11MB | Too large for mobile bundle |
| ViT-Base | 85.8M | ~85MB | Orders of magnitude too large |

MobileNetV2 was specifically designed for mobile inference (depthwise separable convolutions, inverted residuals). An existing fine-tuned banana classifier (`jorgealbert/modelo-mobilenetv2-clasificador-bananas`) confirms the architecture is viable for this task.

The backbone is swappable via the `backbone` argument on `RipenessClassifier` — experimentation with EfficientNet-B0 or other architectures does not require changes elsewhere.

## Consequences

- Inference meets the < 500ms target on CPU-only mid-range devices
- ONNX int8 export produces a ~2.3MB model file, acceptable for APK bundling
- ImageNet pre-training provides strong feature extraction with limited domain data
