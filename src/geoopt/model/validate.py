"""Hold-out-by-hole cross-validation of the ore model (seed data only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from geoopt.config import ModelConfig
from geoopt.model.gp import GPOreModel
from geoopt.model.transforms import to_log_grade

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVReport:
    rmse_log: float
    rmse_log_baseline: float  # predicting each fold's training mean
    coverage_1sd: float  # expected ~0.68
    coverage_2sd: float  # expected ~0.95
    reliability: pd.DataFrame  # predicted P(ore) bin vs observed ore fraction


def cross_validate(
    composites: pd.DataFrame, cfg: ModelConfig, seed: int, n_folds: int | None = None
) -> CVReport:
    """Folds split whole holes, never intervals of one hole across folds."""
    n_folds = n_folds or cfg.cv_folds
    rng = np.random.default_rng(seed)
    holes = composites["hole_id"].unique()
    fold_of = dict(zip(rng.permutation(holes), np.arange(len(holes)) % n_folds, strict=True))
    fold = composites["hole_id"].map(fold_of).to_numpy()
    xyz = composites[["x", "y", "z"]].to_numpy()
    cu = composites["cu"].to_numpy()
    y = to_log_grade(cu, cfg.log_eps)

    mean = np.empty(len(y))
    std = np.empty(len(y))
    p_ore = np.empty(len(y))
    base = np.empty(len(y))
    for k in range(n_folds):
        test = fold == k
        model = GPOreModel(cfg, seed + k)
        model.fit(xyz[~test], cu[~test])
        mean[test], std[test] = model.predict(xyz[test], include_noise=True)
        p_ore[test] = model.prob_ore(xyz[test])
        base[test] = y[~test].mean()

    z = np.abs(y - mean) / std
    is_ore = cu > cfg.cutoff_pct
    bins = pd.cut(p_ore, np.linspace(0, 1, 6), include_lowest=True)
    reliability = (
        pd.DataFrame({"bin": bins, "p": p_ore, "ore": is_ore})
        .groupby("bin", observed=True)
        .agg(n=("p", "size"), mean_pred=("p", "mean"), observed=("ore", "mean"))
        .reset_index()
    )
    report = CVReport(
        rmse_log=float(np.sqrt(np.mean((y - mean) ** 2))),
        rmse_log_baseline=float(np.sqrt(np.mean((y - base) ** 2))),
        coverage_1sd=float(np.mean(z <= 1)),
        coverage_2sd=float(np.mean(z <= 2)),
        reliability=reliability,
    )
    log.info(
        "cv folds=%d rmse_log=%.3f baseline=%.3f cov1=%.2f cov2=%.2f",
        n_folds,
        report.rmse_log,
        report.rmse_log_baseline,
        report.coverage_1sd,
        report.coverage_2sd,
    )
    return report
