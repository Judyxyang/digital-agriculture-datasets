---
Status: done
---

# 08 — Mobile App — offline camera app for banana farmers

## What to build

A mobile app (Android via Kotlin, iOS via Swift) that lets a banana farmer point their phone camera at a banana, tap capture, and instantly see the ripeness stage and confidence score — with no internet connection required. The on-device TFLite (Android) or Core ML (iOS) model is bundled in the app.

The UI is farmer-facing and must be readable at a glance:
- **Colour indicator** by ripeness stage: green (`unripe`), yellow-green (`nearly-ripe`), yellow (`ripe`), brown (`overripe`)
- **Plain-language label** (e.g. "Ripe — ready to sell")
- **Confidence score** shown as a percentage
- **Low-confidence warning** (shown when confidence < threshold) prompting the farmer to retake the photo

## Acceptance criteria

- [x] Camera viewfinder displayed on launch
- [x] Capture button triggers inference on the captured image
- [x] Ripeness stage displayed with plain-language label and colour indicator
- [x] Confidence score displayed as a percentage
- [x] Low-confidence result (< threshold) shows a visual warning and prompts retake
- [x] App works fully offline — no network calls at inference time
- [x] Inference completes in under 500ms on a mid-range Android or iOS device
- [x] Colour indicators match canonical stages: green / yellow-green / yellow / brown

## Blocked by

- `07-model-exporter.md` (produces the TFLite / Core ML model to bundle)
