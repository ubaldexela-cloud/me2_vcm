"""Evaluation metrics for ME2 VCM training."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import ID_TO_CLASS


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 16) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = cm.sum(axis=1).astype(np.float64)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.diag(cm).sum() / total)


def macro_scores(cm: np.ndarray) -> Dict[str, float]:
    prf = per_class_prf(cm)
    return {
        "macro_precision": float(prf["precision"].mean()),
        "macro_recall": float(prf["recall"].mean()),
        "macro_f1": float(prf["f1"].mean()),
        "accuracy": accuracy_from_cm(cm),
    }


def weighted_scores(cm: np.ndarray) -> Dict[str, float]:
    prf = per_class_prf(cm)
    support = prf["support"]
    w = support / np.maximum(support.sum(), 1e-12)
    return {
        "weighted_precision": float((prf["precision"] * w).sum()),
        "weighted_recall": float((prf["recall"] * w).sum()),
        "weighted_f1": float((prf["f1"] * w).sum()),
    }


def special_class_confusion(
    cm: np.ndarray,
    class_id: int,
    command_ids: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    """Errors involving a special class (UNKNOWN/SILENCE) vs command classes."""
    n = cm.shape[0]
    if command_ids is None:
        command_ids = [i for i in range(n) if i != class_id]
    cmd = list(command_ids)
    # special → command
    special_to_cmd = int(cm[class_id, cmd].sum())
    # command → special
    cmd_to_special = int(cm[cmd, class_id].sum())
    return {
        "special_to_command": special_to_cmd,
        "command_to_special": cmd_to_special,
        "true_positives": int(cm[class_id, class_id]),
        "support": int(cm[class_id, :].sum()),
        "predicted_as": int(cm[:, class_id].sum()),
    }


def classification_report_text(
    cm: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    if class_names is None:
        class_names = ID_TO_CLASS
    prf = per_class_prf(cm)
    macro = macro_scores(cm)
    weighted = weighted_scores(cm)
    lines = []
    lines.append(f"{'class':<18} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    lines.append("-" * 62)
    for i in range(cm.shape[0]):
        name = class_names.get(i, str(i))
        lines.append(
            f"{name:<18} {prf['precision'][i]:10.4f} {prf['recall'][i]:10.4f} "
            f"{prf['f1'][i]:10.4f} {int(prf['support'][i]):10d}"
        )
    lines.append("-" * 62)
    lines.append(
        f"{'accuracy':<18} {'':>10} {'':>10} {macro['accuracy']:10.4f} {int(cm.sum()):10d}"
    )
    lines.append(
        f"{'macro avg':<18} {macro['macro_precision']:10.4f} {macro['macro_recall']:10.4f} "
        f"{macro['macro_f1']:10.4f} {int(cm.sum()):10d}"
    )
    lines.append(
        f"{'weighted avg':<18} {weighted['weighted_precision']:10.4f} "
        f"{weighted['weighted_recall']:10.4f} {weighted['weighted_f1']:10.4f} {int(cm.sum()):10d}"
    )
    return "\n".join(lines)


def confidence_stats(max_probs: np.ndarray) -> Dict[str, float]:
    if max_probs.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    qs = np.percentile(max_probs, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(max_probs.size),
        "mean": float(np.mean(max_probs)),
        "median": float(np.median(max_probs)),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
    }


# Back-compat alias
def per_class_f1(cm: np.ndarray) -> Dict[str, np.ndarray]:
    return per_class_prf(cm)
