---
Status: done
---

# 05 — TrainingPipeline — end-to-end train loop with checkpoint saving

## What to build

Implement a `TrainingPipeline` that orchestrates the full offline training workflow: load dataset → preprocess → train RipenessClassifier → evaluate → save best checkpoint. Used offline only; not deployed to edge.

## Acceptance criteria

- [x] Orchestrates DatasetLoader → ImagePreprocessor → RipenessClassifier training loop
- [x] Evaluates on the validation split after each epoch using the Evaluator
- [x] Saves the best checkpoint based on validation F1
- [x] Training run is reproducible given the same dataset and random seed
- [x] Produces a checkpoint file that InferencePipeline can load

## Blocked by

- `01-dataset-loader.md`
- `02-image-preprocessor.md`
- `03-ripeness-classifier.md`
- `04-evaluator.md`
