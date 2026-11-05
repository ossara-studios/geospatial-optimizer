"""Hole-level acquisition: aggregate point predictions along each candidate hole."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geoopt.config import PlannerConfig

SCORE_COLUMNS = [
    "hole_idx",
    "exp_ore_m",
    "explore",
    "exp_ore_per_m",
    "explore_per_m",
    "boundary_per_m",
    "delin_per_m",
    "spacing_penalty",
    "score",
]


def _zscore(v: np.ndarray) -> np.ndarray:
    std = v.std()
    return (v - v.mean()) / std if std > 0 else np.zeros_like(v)


def spacing_penalty(
    collar_xy: np.ndarray, drilled_xy: np.ndarray, min_spacing_m: float, weight: float
) -> np.ndarray:
    """Linear penalty from `weight` at distance 0 to 0 at `min_spacing_m`."""
    if len(drilled_xy) == 0 or min_spacing_m == 0 or weight == 0:
        return np.zeros(len(collar_xy))
    d = np.linalg.norm(collar_xy[:, None, :] - drilled_xy[None, :, :], axis=-1).min(axis=1)
    return weight * np.clip(1.0 - d / min_spacing_m, 0.0, None)


def score_candidates(
    candidates: pd.DataFrame,
    points: pd.DataFrame,
    p_ore: np.ndarray,
    std: np.ndarray,
    drilled_xy: np.ndarray,
    cfg: PlannerConfig,
    p_latent: np.ndarray | None = None,
    delin: pd.Series | None = None,
) -> pd.DataFrame:
    """Score candidate holes.

    `points` has one row per sample location (hole_idx, length_m) aligned with `p_ore`
    (P(Cu > cutoff)) and `std` (latent log-grade std). Per hole:
        exp_ore_m = sum(P(ore) * l), explore = sum(sigma * l), both divided by depth,
    then optionally z-scored across candidates before score = ore + beta * explore - penalty.
    With `p_latent` (latent P(ore)), boundary = sum(p * (1 - p) * l) / depth adds gamma * boundary.
    With `delin` (targeted variance reduction per hole_idx), delin / depth adds delta * delin.
    The expected-ore term is weighted by ore_weight (0 = pure delineation).
    """
    per_point = pd.DataFrame(
        {
            "hole_idx": points["hole_idx"].to_numpy(),
            "ore_m": p_ore * points["length_m"].to_numpy(),
            "sigma_m": std * points["length_m"].to_numpy(),
            "boundary_m": (
                p_latent * (1 - p_latent) * points["length_m"].to_numpy()
                if p_latent is not None
                else 0.0
            ),
        }
    )
    agg = per_point.groupby("hole_idx").sum()
    out = candidates[["hole_idx", "x", "y", "depth"]].merge(
        agg, left_on="hole_idx", right_index=True, how="inner"
    )
    out["exp_ore_m"] = out.pop("ore_m")
    out["explore"] = out.pop("sigma_m")
    out["exp_ore_per_m"] = out["exp_ore_m"] / out["depth"]
    out["explore_per_m"] = out["explore"] / out["depth"]
    out["boundary_per_m"] = out.pop("boundary_m") / out["depth"]
    out["delin_per_m"] = (
        out["hole_idx"].map(delin).fillna(0.0) / out["depth"] if delin is not None else 0.0
    )
    ore = out["exp_ore_per_m"].to_numpy()
    explore = out["explore_per_m"].to_numpy()
    boundary = out["boundary_per_m"].to_numpy()
    delin_pm = out["delin_per_m"].to_numpy(dtype=float)
    if cfg.standardize:
        ore, explore, boundary = _zscore(ore), _zscore(explore), _zscore(boundary)
        delin_pm = _zscore(delin_pm)
    out["spacing_penalty"] = spacing_penalty(
        out[["x", "y"]].to_numpy(), drilled_xy, cfg.min_spacing_m, cfg.spacing_weight
    )
    out["score"] = (
        cfg.ore_weight * ore
        + cfg.beta * explore
        + cfg.gamma * boundary
        + cfg.delta * delin_pm
        - out["spacing_penalty"]
    )
    return out[SCORE_COLUMNS].sort_values("score", ascending=False).reset_index(drop=True)
