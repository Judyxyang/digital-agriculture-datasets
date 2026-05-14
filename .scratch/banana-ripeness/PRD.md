---
Status: ready-for-agent
---

# PRD: AI Detection of Banana Ripeness

## Problem Statement

Agricultural workers, distributors, and retailers currently assess banana ripeness stage by eye — a process that is slow, inconsistent across individuals, and impractical at scale. There is no reliable, automated way to classify whether a banana is `unripe`, `nearly-ripe`, `ripe`, or `overripe` from an image, especially under the varied lighting conditions of field, warehouse, and market environments.

## Solution

A computer vision system that accepts an image of a banana and returns its ripeness stage (`unripe`, `nearly-ripe`, `ripe`, or `overripe`) along with a confidence score. The system is designed to run on modest hardware (edge devices / mobile) with an inference latency target of under 500ms per image. It is trained on a balanced dataset of banana images covering all four ripeness stages and varied lighting conditions.

## User Stories

1. As a warehouse operator, I want to photograph a banana and receive its ripeness stage instantly, so that I can sort batches without manual inspection.
2. As a field agricultural worker, I want the system to work under natural outdoor lighting, so that I can assess ripeness on the farm without controlled conditions.
3. As a retailer, I want to detect overripe bananas automatically, so that I can remove them from display before they affect customer perception.
4. As a dataset curator, I want to load a directory of labeled banana images and get balanced train/val/test splits, so that model training is reproducible.
5. As a dataset curator, I want augmentation applied automatically during training (brightness/contrast jitter), so that the model generalises across lighting conditions.
6. As an ML engineer, I want to train a ripeness classifier on a labeled dataset, so that I can produce a model checkpoint ready for deployment.
7. As an ML engineer, I want to evaluate per-class precision, recall, and F1 across all four ripeness stages, so that I can identify which stages the model confuses before deploying.
8. As an ML engineer, I want the model to use a lightweight backbone (e.g. MobileNetV2), so that inference stays within the 500ms latency budget on edge hardware.
9. As a developer integrating the system, I want a single `predict(image) → (ripeness_stage, confidence_score)` call, so that I can integrate ripeness detection without understanding model internals.
10. As a developer, I want the InferencePipeline to load a checkpoint and be ready to predict with minimal setup, so that deployment to mobile/edge is straightforward.
11. As a developer, I want the ImagePreprocessor to handle resizing, normalisation, and ROI extraction consistently between training and inference, so that there is no train/serve skew.
12. As a quality reviewer, I want confidence scores returned alongside each ripeness stage prediction, so that I can flag low-confidence results for human review.
13. As a system operator, I want inference to complete in under 500ms per image, so that the system is usable in real-time sorting workflows.
14. As a researcher, I want to swap the classifier backbone without changing the rest of the pipeline, so that I can experiment with newer architectures easily.
15. As a dataset curator, I want the dataset to be balanced across the four ripeness stages, so that the model does not develop a bias toward the most common stage.

## Implementation Decisions

### Modules

**DatasetLoader**
- Loads labeled banana images from a root directory structured by ripeness stage class
- Applies augmentation during training (random brightness/contrast jitter, horizontal flip)
- Returns balanced train / val / test splits as iterable DataLoaders
- Interface: `load(path, split) → DataLoader`

**ImagePreprocessor**
- Shared preprocessing used by both training and inference paths (resize, normalise, ROI extraction)
- Ensures no train/serve skew — single source of truth for image transformations
- Interface: `preprocess(raw_image) → tensor`

**RipenessClassifier**
- Wraps a pretrained MobileNetV2 backbone fine-tuned on the 4-class ripeness scheme
- Loads weights from a checkpoint file at construction time
- Interface: `predict(image) → (ripeness_stage, confidence_score)`
- `ripeness_stage` is one of: `unripe`, `nearly-ripe`, `ripe`, `overripe`
- `confidence_score` is the softmax probability for the top predicted class

**TrainingPipeline**
- Orchestrates: DatasetLoader → ImagePreprocessor → RipenessClassifier training loop → checkpoint saving
- Used offline only; not deployed to edge
- Saves best checkpoint based on validation F1

**InferencePipeline**
- Lightweight runtime path: loads checkpoint → ImagePreprocessor → RipenessClassifier.predict
- Accepts a raw image input, returns ripeness stage + confidence score
- Must meet the < 500ms latency budget

**Evaluator**
- Computes per-class precision, recall, and F1 for all four ripeness stages
- Accepts model predictions and ground-truth labels
- Interface: `evaluate(predictions, labels) → MetricsReport`

### Architectural Decisions

- MobileNetV2 chosen as backbone for edge/mobile compatibility; architecture is swappable via config
- ImagePreprocessor is shared between training and inference to eliminate train/serve skew
- Dataset directory structure follows the standard `<root>/<class>/image.jpg` convention for compatibility with common dataset tools
- Confidence score is softmax probability of top class; callers may apply their own threshold for human-review escalation

## Testing Decisions

### What makes a good test

Test external behaviour only — what goes in and what comes out. Do not assert on internal state, private methods, or implementation details. A test should remain valid even if the internal implementation is completely rewritten.

### Modules to test

| Module | What to test |
|---|---|
| `DatasetLoader` | Returns correct split sizes; augmentation applied only to train split; output is balanced across classes |
| `ImagePreprocessor` | Output tensor shape and dtype are correct; pixel values are normalised to expected range; ROI extraction produces expected crop dimensions |
| `RipenessClassifier` | `predict()` returns a valid ripeness stage string and a confidence score in [0, 1]; handles a batch of images; output stage is one of the four canonical classes |
| `Evaluator` | Metrics are correct for known predictions vs labels; handles edge cases (all-correct, all-wrong, missing class in predictions) |

### Prior art

No existing tests in this repo yet. Follow standard pytest conventions; use small synthetic images (e.g. solid-colour numpy arrays) rather than real image files in unit tests to keep the suite fast and dependency-free.

## Out of Scope

- Real-time video stream processing (single image inference only)
- Detection of multiple bananas in one image (single ROI per image)
- Disease or defect detection beyond ripeness stage
- Deployment tooling (containerisation, model serving APIs)
- Dataset collection or labeling tooling
- Support for fruit other than bananas

## Further Notes

- The dataset must be balanced across all four ripeness stages before training begins; the DatasetLoader should enforce or warn on imbalance
- Low-confidence predictions (threshold to be determined by the operator) should be surfaced for human review rather than acted on automatically
- All ripeness stage strings in code, issues, and tests must match the canonical 4-class vocabulary exactly: `unripe`, `nearly-ripe`, `ripe`, `overripe`
