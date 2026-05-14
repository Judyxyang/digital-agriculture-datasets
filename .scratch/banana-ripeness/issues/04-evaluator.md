---
Status: ready-for-agent
---

# 04 — Evaluator — per-class precision, recall, F1 report

## What to build

Implement an `Evaluator` that computes per-class precision, recall, and F1 score across all four ripeness stages. Used after training to validate model quality before deployment.

## Acceptance criteria

- [ ] Accepts model predictions and ground-truth labels
- [ ] Returns per-class precision, recall, and F1 for all four ripeness stages
- [ ] Interface: `evaluate(predictions, labels) → MetricsReport`
- [ ] Unit tests pass: metrics are correct for known prediction/label pairs; handles edge cases — all-correct, all-wrong, a ripeness stage missing from predictions

## Blocked by

None — can start immediately
