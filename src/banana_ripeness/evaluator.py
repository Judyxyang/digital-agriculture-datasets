from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

RIPENESS_STAGES = ("unripe", "nearly-ripe", "ripe", "overripe")


@dataclass(frozen=True)
class ClassMetrics:
    stage: str
    precision: float
    recall: float
    f1: float
    support: int   # number of true instances


@dataclass(frozen=True)
class MetricsReport:
    per_class: List[ClassMetrics]
    macro_f1: float

    def __str__(self) -> str:
        lines = ["Ripeness Stage    Precision  Recall  F1     Support"]
        lines.append("-" * 52)
        for m in self.per_class:
            lines.append(
                f"{m.stage:<18}{m.precision:.3f}      {m.recall:.3f}   {m.f1:.3f}  {m.support}"
            )
        lines.append(f"\nMacro F1: {self.macro_f1:.3f}")
        return "\n".join(lines)


class Evaluator:
    """Computes per-class precision, recall, and F1 across all ripeness stages.

    Interface: evaluator.evaluate(predictions, labels) → MetricsReport

    Args:
        predictions: Sequence of predicted class indices (int) or stage
                     strings. Mixed types are not supported.
        labels:      Sequence of ground-truth class indices (int) or stage
                     strings, matching the type of predictions.
    """

    def evaluate(
        self,
        predictions: Sequence[int | str],
        labels: Sequence[int | str],
    ) -> MetricsReport:
        if len(predictions) != len(labels):
            raise ValueError(
                f"predictions and labels must have the same length, "
                f"got {len(predictions)} and {len(labels)}"
            )
        if len(predictions) == 0:
            raise ValueError("predictions and labels must not be empty")

        preds = [self._to_idx(p) for p in predictions]
        trues = [self._to_idx(t) for t in labels]

        n = len(RIPENESS_STAGES)
        tp = [0] * n
        fp = [0] * n
        fn = [0] * n

        for p, t in zip(preds, trues):
            if p == t:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        per_class = []
        f1_scores = []
        for i, stage in enumerate(RIPENESS_STAGES):
            support = tp[i] + fn[i]
            precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class.append(ClassMetrics(stage, precision, recall, f1, support))
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / n
        return MetricsReport(per_class=per_class, macro_f1=macro_f1)

    @staticmethod
    def _to_idx(value: int | str) -> int:
        if isinstance(value, int):
            if not (0 <= value < len(RIPENESS_STAGES)):
                raise ValueError(
                    f"Class index {value} out of range [0, {len(RIPENESS_STAGES)})"
                )
            return value
        if isinstance(value, str):
            if value not in RIPENESS_STAGES:
                raise ValueError(
                    f"Unknown ripeness stage '{value}'. "
                    f"Must be one of {RIPENESS_STAGES}"
                )
            return RIPENESS_STAGES.index(value)
        raise TypeError(f"Expected int or str, got {type(value).__name__}")
