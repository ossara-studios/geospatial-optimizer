from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoopt.config import PlannerConfig
from geoopt.planner.acquisition import score_candidates, spacing_penalty
from geoopt.planner.strategies import (
    AdaptiveStrategy,
    GridStrategy,
    HistoricalStrategy,
    PlannerState,
    RandomStrategy,
)

CFG = PlannerConfig(
    beta=0.5, standardize=True, min_spacing_m=50.0, spacing_weight=1.0, grid_spacing_m=100.0
)


def _toy(n: int = 6, depth: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = pd.DataFrame(
        {
            "hole_idx": np.arange(n),
            "hole_id": [f"C{i}" for i in range(n)],
            "x": np.arange(n) * 200.0,
            "y": np.zeros(n),
            "z": np.zeros(n),
            "azimuth": 0.0,
            "dip": -90.0,
            "depth": depth,
            "drill_date": pd.date_range("2010-01-01", periods=n, freq="MS")[::-1],
        }
    )
    pts = pd.DataFrame(
        [
            (h, 10.0, x, 0.0, -d)
            for h, x in zip(cand["hole_idx"], cand["x"], strict=True)
            for d in np.arange(5, depth, 10)
        ],
        columns=["hole_idx", "length_m", "x", "y", "z"],
    )
    return cand, pts


class FakeModel:
    """P(ore) and sigma set per hole by its collar x."""

    def __init__(self, p_by_x: dict[float, float], s_by_x: dict[float, float]) -> None:
        self.p, self.s = p_by_x, s_by_x

    def prob_ore(
        self, xyz: np.ndarray, cutoff_pct: float | None = None, include_noise: bool = True
    ) -> np.ndarray:
        return np.array([self.p[x] for x in xyz[:, 0]])

    def predict(self, xyz: np.ndarray, include_noise: bool = False):
        return np.zeros(len(xyz)), np.array([self.s[x] for x in xyz[:, 0]])


def _state(cand, pts, model=None, available=None, drilled=None) -> PlannerState:
    return PlannerState(
        candidates=cand,
        points=pts,
        available=set(cand["hole_idx"]) if available is None else available,
        drilled_xy=np.empty((0, 2)) if drilled is None else drilled,
        model=model,
    )


def test_adaptive_prefers_high_p_ore_then_high_sigma() -> None:
    cand, pts = _toy()
    xs = cand["x"].tolist()
    p = dict(zip(xs, [0.1, 0.9, 0.1, 0.1, 0.1, 0.1], strict=True))
    s = dict(zip(xs, [0.5, 0.5, 0.5, 2.0, 0.5, 0.5], strict=True))
    strat = AdaptiveStrategy(CFG)
    assert strat.recommend(_state(cand, pts, FakeModel(p, s))).hole_idx == 1
    # With hole 1 excluded, the high-uncertainty hole wins.
    assert strat.recommend(_state(cand, pts, FakeModel(p, s), {0, 2, 3, 4, 5})).hole_idx == 3
    # Pure exploitation ignores sigma.
    greedy = AdaptiveStrategy(CFG.model_copy(update={"beta": 0.0}))
    assert greedy.recommend(_state(cand, pts, FakeModel(p, s), {0, 2, 3, 4, 5})).hole_idx in {
        0,
        2,
        4,
        5,
    }


def test_score_is_per_metre() -> None:
    cand, pts = _toy(n=2)
    cand.loc[1, "depth"] = 300.0
    long_pts = pts[pts["hole_idx"] == 1]
    extra = pd.concat(
        [long_pts.assign(z=long_pts["z"] - 100 * k) for k in (1, 2)], ignore_index=True
    )
    pts = pd.concat([pts, extra], ignore_index=True)
    out = score_candidates(
        cand, pts, np.full(len(pts), 0.5), np.ones(len(pts)), np.empty((0, 2)), CFG
    )
    assert out["exp_ore_per_m"].tolist() == pytest.approx([0.5, 0.5])
    assert out.set_index("hole_idx").loc[1, "exp_ore_m"] == pytest.approx(150.0)


def test_spacing_penalty() -> None:
    pen = spacing_penalty(
        np.array([[0.0, 0.0], [25.0, 0.0], [80.0, 0.0]]), np.array([[0.0, 0.0]]), 50.0, 2.0
    )
    np.testing.assert_allclose(pen, [2.0, 1.0, 0.0])


def test_adaptive_avoids_holes_next_to_drilled_ones() -> None:
    cand, pts = _toy()
    xs = cand["x"].tolist()
    p = dict(zip(xs, [0.5, 0.52, 0.1, 0.1, 0.1, 0.1], strict=True))
    s = {x: 0.5 for x in xs}
    drilled = np.array([[200.0, 0.0]])  # right on top of hole 1
    strat = AdaptiveStrategy(CFG.model_copy(update={"spacing_weight": 10.0}))
    assert strat.recommend(_state(cand, pts, FakeModel(p, s), drilled=drilled)).hole_idx == 0


def test_random_is_reproducible_and_respects_availability() -> None:
    cand, pts = _toy()
    picks = [
        RandomStrategy(3).recommend(_state(cand, pts, available={2, 4})).hole_idx for _ in range(5)
    ]
    assert set(picks) <= {2, 4}
    assert (
        RandomStrategy(3).recommend(_state(cand, pts)).hole_idx
        == RandomStrategy(3).recommend(_state(cand, pts)).hole_idx
    )


def test_grid_visits_in_order_and_skips_unavailable() -> None:
    cand, pts = _toy()
    grid = GridStrategy(cand, spacing_m=200.0)
    available = set(cand["hole_idx"]) - {1}
    order = []
    for _ in range(len(available)):
        pick = grid.recommend(_state(cand, pts, available=available)).hole_idx
        order.append(pick)
        available.discard(pick)
    assert order == [0, 2, 3, 4, 5]


def test_historical_follows_drill_dates() -> None:
    cand, pts = _toy()
    strat = HistoricalStrategy(cand)
    assert strat.recommend(_state(cand, pts)).hole_idx == 5  # dates are reversed
    assert strat.recommend(_state(cand, pts, available={0, 1})).hole_idx == 1


def test_no_candidates_left() -> None:
    cand, pts = _toy()
    with pytest.raises(LookupError):
        RandomStrategy(0).recommend(_state(cand, pts, available=set()))


def test_boundary_term_prefers_holes_through_ore_waste_edges() -> None:
    cand, pts = _toy()
    xs = cand["x"].tolist()
    p = dict(zip(xs, [0.5, 0.05, 0.95, 0.05, 0.05, 0.05], strict=True))
    s = {x: 0.5 for x in xs}
    pts_x = pts["x"].to_numpy()
    p_latent = np.array([p[x] for x in pts_x])
    no_gamma = score_candidates(cand, pts, p_latent, np.full(len(pts), 0.5), np.empty((0, 2)), CFG)
    assert no_gamma.iloc[0]["hole_idx"] == 2  # most ore wins without the boundary term
    edge = CFG.model_copy(update={"gamma": 4.0})
    out = score_candidates(
        cand, pts, p_latent, np.full(len(pts), 0.5), np.empty((0, 2)), edge, p_latent
    )
    assert out.iloc[0]["hole_idx"] == 0  # P = 0.5 maximises p(1-p)
    assert out.set_index("hole_idx").loc[0, "boundary_per_m"] == pytest.approx(0.25)
    strat = AdaptiveStrategy(edge)
    assert strat.recommend(_state(cand, pts, FakeModel(p, s))).hole_idx == 0
