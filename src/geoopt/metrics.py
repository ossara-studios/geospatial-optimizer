"""Backtest metrics, all computed against real held-out assays (never a synthetic truth)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ore_metres(composites: pd.DataFrame, cutoff_pct: float) -> float:
    """Composite metres above cutoff (discovery)."""
    return float(composites.loc[composites["cu"] > cutoff_pct, "length_m"].sum())


def classification(predicted_ore: np.ndarray, actual_ore: np.ndarray) -> dict[str, float]:
    """F1 and balanced accuracy of an ore/waste call."""
    p, a = np.asarray(predicted_ore, bool), np.asarray(actual_ore, bool)
    tp, fp = np.sum(p & a), np.sum(p & ~a)
    fn, tn = np.sum(~p & a), np.sum(~p & ~a)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "n": int(len(p)),
    }


def metres_to_target(curve: pd.DataFrame, column: str, target: float) -> float:
    """First cumulative metres at which `column` reaches `target` (NaN if never)."""
    hit = curve.loc[curve[column] >= target, "cum_metres"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def on_metre_grid(curve: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    """Step-interpolate a metric onto common cumulative-metre values (value after each hole)."""
    x = curve["cum_metres"].to_numpy()
    y = curve[column].to_numpy()
    idx = np.searchsorted(x, grid, side="right") - 1
    out = np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)
    out[grid > x.max()] = np.nan  # don't extrapolate past the run's end
    return out
