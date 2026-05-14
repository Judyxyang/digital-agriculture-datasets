# ADR-0003 — ONNX as the mobile export format

**Date:** 2026-05-14  
**Status:** Accepted

## Context

The trained PyTorch model must be converted to a format that runs on Android (and optionally iOS) without internet access. Three formats were considered: TFLite, Core ML, and ONNX.

## Decision

Export to **ONNX with int8 dynamic quantization**, served via **ONNX Runtime Mobile** on Android.

## Rationale

| Format | Android | iOS | Requires | Notes |
|---|---|---|---|---|
| TFLite | ✓ | ✗ | `tensorflow` | Android-only; adds ~100MB TF dependency to install |
| Core ML | ✗ | ✓ | `coremltools` + macOS | iOS-only |
| **ONNX** | ✓ | ✓ | `onnxruntime` | Cross-platform; direct PyTorch export |

ONNX Runtime Mobile (`onnxruntime-android`, `onnxruntime-objc`) works on both platforms, avoiding format fragmentation. The PyTorch → ONNX export path is first-class and actively maintained.

int8 dynamic quantization reduces model weights to 8-bit integers at export time. The `ModelExporter` applies `quantize_dynamic` and validates that quantized outputs match FP32 outputs within `atol=1e-4`.

**Note on file size:** ONNX int8 MobileNetV2 is ~2.3MB. TFLite achieves ~600KB due to a more compact format. The 2.3MB ONNX file is acceptable for APK bundling (typical app size is 10–50MB).

## Consequences

- `model_int8.onnx` is copied to `mobile/android/app/src/main/assets/` and bundled in the APK
- No network call required at inference time — model is fully on-device
- TFLite and Core ML export stubs are provided in `ModelExporter` with helpful `ImportError` messages should they be needed in future
- If iOS deployment is prioritised, switch to Core ML: `ModelExporter().export(..., format="coreml")` after `pip install coremltools`
