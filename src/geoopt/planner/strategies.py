"""Strategies that pick the next candidate hole. All see the same candidate pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from geoopt.config import PlannerConfig
from geoopt.model.gp import OreModel
from geoopt.planner.acquisition import score_candidates


@dataclass
class PlannerState:
    candidates: pd.DataFrame  # hole_idx, hole_id, x, y, z, azimuth, dip, depth[, drill_date]
    points: pd.DataFrame  # hole_idx, length_m, x, y, z (geometry only)
    available: set[int]  # hole_idx not yet drilled, rejected or denied
    drilled_xy: np.ndarray  # collars of every drilled hole (seed + loop), shape (n, 2)
    model: OreModel | None = None


@dataclass(frozen=True)
class Recommendation:
    hole_idx: int
    score: float
    exp_ore_m: float
    prob_ore: float  # mean P(ore) along the hole
    breakdown: dict[str, float]


class Strategy(Protocol):
    name: str

    def recommend(self, state: PlannerState) -> Recommendation: ...


def _flat(hole_idx: int) -> Recommendation:
    return Recommendation(hole_idx, float("nan"), float("nan"), float("nan"), {})


def _require_available(state: PlannerState) -> None:
    if not state.available:
        raise LookupError("no candidate holes left")


class AdaptiveStrategy:
    name = "adaptive"

    def __init__(self, cfg: PlannerConfig) -> None:
        self._cfg = cfg

    def recommend(self, state: PlannerState) -> Recommendation:
        _require_available(state)
        if state.model is None:
            raise ValueError("adaptive strategy needs a fitted model")
        cand = state.candidates[state.candidates["hole_idx"].isin(state.available)]
        pts = state.points[state.points["hole_idx"].isin(state.available)]
        xyz = pts[["x", "y", "z"]].to_numpy()
        p_ore = state.model.prob_ore(xyz)
        _, std = state.model.predict(xyz)
        scores = score_candidates(cand, pts, p_ore, std, state.drilled_xy, self._cfg)
        best = scores.iloc[0]
        depth = float(cand.loc[cand["hole_idx"] == best["hole_idx"], "depth"].iloc[0])
        return Recommendation(
            hole_idx=int(best["hole_idx"]),
            score=float(best["score"]),
            exp_ore_m=float(best["exp_ore_m"]),
            prob_ore=float(best["exp_ore_m"] / depth),
            breakdown={
                k: float(best[k])
                for k in ("exp_ore_per_m", "explore_per_m", "explore", "spacing_penalty")
            },
        )


class RandomStrategy:
    name = "random"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def recommend(self, state: PlannerState) -> Recommendation:
        _require_available(state)
        return _flat(int(self._rng.choice(sorted(state.available))))


class GridStrategy:
    """Visit the available hole nearest each node of a regular grid, in row-major order."""

    name = "grid"

    def __init__(self, candidates: pd.DataFrame, spacing_m: float) -> None:
        xy = candidates[["x", "y"]].to_numpy()
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        gx = np.arange(lo[0], hi[0] + spacing_m, spacing_m)
        gy = np.arange(lo[1], hi[1] + spacing_m, spacing_m)
        self._nodes = np.array([(x, y) for y in gy for x in gx])
        self._xy = xy
        self._idx = candidates["hole_idx"].to_numpy()
        self._spacing = spacing_m
        self._next = 0

    def recommend(self, state: PlannerState) -> Recommendation:
        _require_available(state)
        avail = np.isin(self._idx, list(state.available))
        while self._next < len(self._nodes):
            node = self._nodes[self._next]
            self._next += 1
            d = np.linalg.norm(self._xy - node, axis=1)
            d[~avail] = np.inf
            # Only nodes with a hole inside their own cell; empty cells are skipped.
            if d.min() <= self._spacing * np.sqrt(2) / 2:
                return _flat(int(self._idx[np.argmin(d)]))
        # Grid exhausted: fall back to the remaining holes in hole_idx order.
        return _flat(min(state.available))


class HistoricalStrategy:
    """The order the holes were actually drilled (needs drill_date)."""

    name = "historical"

    def __init__(self, candidates: pd.DataFrame) -> None:
        if "drill_date" not in candidates or candidates["drill_date"].isna().any():
            raise ValueError("historical strategy needs a drill_date on every candidate")
        self._order = candidates.sort_values(["drill_date", "hole_idx"])["hole_idx"].tolist()

    def recommend(self, state: PlannerState) -> Recommendation:
        _require_available(state)
        return _flat(next(h for h in self._order if h in state.available))


def make_strategy(name: str, cfg: PlannerConfig, candidates: pd.DataFrame, seed: int) -> Strategy:
    if name == "adaptive":
        return AdaptiveStrategy(cfg)
    if name == "random":
        return RandomStrategy(seed)
    if name == "grid":
        return GridStrategy(candidates, cfg.grid_spacing_m)
    if name == "historical":
        return HistoricalStrategy(candidates)
    raise ValueError(f"unknown strategy {name!r}")
