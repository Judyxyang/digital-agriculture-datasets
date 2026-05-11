"""
Model assessment stage — mirrors the evaluation metrics from the DeepWeeds paper
(Olsen et al., 2019, Scientific Reports).
"""

import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score


class ClassifierAssessment:
    def __init__(self, class_names: List[str], output_dir: str):
        self.class_names = class_names
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, dataset, device, batch_size: int = 32) -> Dict:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
        return self._compute_metrics(np.array(all_labels), np.array(all_preds))

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        n = len(self.class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        per_class = {}
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            support   = int(cm[i, :].sum())
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            per_class[name] = {
                "top1_accuracy": round(recall * 100, 1),
                "precision":     round(precision * 100, 1),
                "fpr":           round(fpr * 100, 2),
                "f1":            round(f1 * 100, 1),
                "support":       support,
            }
        supports = np.array([per_class[n]["support"] for n in self.class_names])
        total    = supports.sum()
        def _wavg(key):
            vals = np.array([per_class[n][key] for n in self.class_names])
            return round(float((vals * supports).sum() / total), 1) if total else 0.0
        return {
            "per_class":        per_class,
            "weighted_avg": {
                "top1_accuracy": _wavg("top1_accuracy"),
                "precision":     _wavg("precision"),
                "fpr":           _wavg("fpr"),
                "f1":            _wavg("f1"),
                "support":       int(total),
            },
            "overall_accuracy": round(accuracy_score(y_true, y_pred) * 100, 1),
            "confusion_matrix": cm.tolist(),
            "num_classes":      n,
        }

    def save_report(self, metrics: Dict, yolo_metrics: Optional[Dict] = None,
                    split_name: str = "test"):
        self._save_text_report(metrics, yolo_metrics, split_name)
        self._save_confusion_matrix(metrics)
        self._save_per_class_chart(metrics)
        self._save_json(metrics, yolo_metrics)
        print(f"\n[Assessment] Reports saved → {self.out}")

    def _save_text_report(self, metrics, yolo_metrics, split_name):
        pc   = metrics["per_class"]
        wavg = metrics["weighted_avg"]
        col  = max(len(n) for n in self.class_names) + 2
        lines = [
            "=" * 80,
            "  WEED CLASSIFIER ASSESSMENT  —  DeepWeeds-style metrics",
            f"  Split: {split_name}   |   Overall accuracy: {metrics['overall_accuracy']:.1f}%",
            "=" * 80, "",
            f"  {'Species':<{col}}  {'Top-1 Acc':>10}  {'Precision':>10}  {'FPR':>8}  {'F1':>8}  {'Support':>8}",
            f"  {'-'*col}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}",
        ]
        for name in self.class_names:
            m = pc[name]
            lines.append(
                f"  {name:<{col}}  {m['top1_accuracy']:>9.1f}%"
                f"  {m['precision']:>9.1f}%"
                f"  {m['fpr']:>7.2f}%"
                f"  {m['f1']:>7.1f}%"
                f"  {m['support']:>8d}"
            )
        lines += [
            f"  {'-'*col}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}",
            f"  {'Weighted avg':<{col}}  {wavg['top1_accuracy']:>9.1f}%"
            f"  {wavg['precision']:>9.1f}%"
            f"  {wavg['fpr']:>7.2f}%"
            f"  {wavg['f1']:>7.1f}%"
            f"  {wavg['support']:>8d}", "",
        ]
        if yolo_metrics:
            lines += [
                "─" * 80, "  YOLO DETECTION METRICS (validation set)", "─" * 80,
                f"  mAP@0.5       : {yolo_metrics.get('mAP50', 0):.4f}",
                f"  mAP@0.5-0.95  : {yolo_metrics.get('mAP50-95', 0):.4f}",
                f"  Precision     : {yolo_metrics.get('precision', 0):.4f}",
                f"  Recall        : {yolo_metrics.get('recall', 0):.4f}", "",
            ]
        lines += [
            "─" * 80,
            "  Reference (DeepWeeds paper — ResNet-50, 5-fold CV):",
            "  Weighted avg top-1 accuracy : 95.7%",
            "  Weighted avg precision      : 95.7%",
            "  Weighted avg FPR            :  2.04%",
            "=" * 80,
        ]
        text = "\n".join(lines)
        print(text)
        (self.out / "assessment_report.txt").write_text(text)

    def _save_confusion_matrix(self, metrics):
        cm = np.array(metrics["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums * 100
        fig, ax = plt.subplots(figsize=(max(8, len(self.class_names)),
                                        max(6, len(self.class_names) - 1)))
        sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    linewidths=0.5, linecolor="#dddddd",
                    cbar_kws={"label": "Row-normalised (%)"}, ax=ax)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_title("Confusion Matrix (row-normalised %)", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(rotation=40, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        out_path = str(self.out / "confusion_matrix.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Assessment] Confusion matrix → {out_path}")

    def _save_per_class_chart(self, metrics):
        pc    = metrics["per_class"]
        names = self.class_names
        n     = len(names)
        x     = np.arange(n)
        width = 0.22
        top1  = [pc[nm]["top1_accuracy"] for nm in names]
        prec  = [pc[nm]["precision"]     for nm in names]
        f1    = [pc[nm]["f1"]            for nm in names]
        fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
        ax.bar(x - width, top1, width, label="Top-1 Accuracy", color="#4c8eda")
        ax.bar(x,         prec, width, label="Precision",       color="#57c27a")
        ax.bar(x + width, f1,   width, label="F1-Score",        color="#e07b54")
        ax.axhline(95.7, color="red", linestyle="--", linewidth=1,
                   label="Paper weighted avg (95.7%)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_title("Per-class Classifier Metrics vs. DeepWeeds Baseline",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = str(self.out / "per_class_metrics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Assessment] Per-class chart → {out_path}")

    def _save_json(self, metrics, yolo_metrics):
        data = dict(metrics)
        if yolo_metrics:
            data["yolo_metrics"] = yolo_metrics
        (self.out / "metrics.json").write_text(json.dumps(data, indent=2))
        print(f"  [Assessment] metrics.json → {self.out / 'metrics.json'}")
