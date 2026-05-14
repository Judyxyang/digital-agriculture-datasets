---
Status: done
---

# 01 — DatasetLoader with balanced splits

## What to build

Implement a `DatasetLoader` that reads labeled banana images from a root directory structured by ripeness stage class (`unripe`, `nearly-ripe`, `ripe`, `overripe`). It must return balanced train / val / test DataLoaders and apply augmentation (random brightness/contrast jitter, horizontal flip) to the training split only.

## Acceptance criteria

- [x] Loads images from `<root>/<ripeness-stage>/image.jpg` directory structure
- [x] Returns train / val / test splits as iterable DataLoaders
- [x] Augmentation (brightness/contrast jitter, horizontal flip) applied to train split only
- [x] Warns or errors if dataset is not balanced across the four ripeness stages
- [x] Interface: `load(path, split) → DataLoader`
- [x] Unit tests pass: correct split sizes; augmentation only on train; output balanced across classes

## Blocked by

None — can start immediately
