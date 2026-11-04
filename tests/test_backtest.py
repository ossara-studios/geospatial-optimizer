from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoopt.backtest import run_backtest, run_beta_sweep
from geoopt.config import Config, SeedSet
from geoopt.metrics import classification, metres_to_target, on_metre_grid, ore_metres


def test_classification() -> None:
    m = classification(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert m["f1"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert classification(np.array([0, 0]), np.array([0, 0]))["f1"] == 0.0


def test_curve_helpers() -> None:
    curve = pd.DataFrame({"cum_metres": [0.0, 100.0, 250.0], "f1": [0.1, 0.5, 0.7]})
    assert metres_to_target(curve, "f1", 0.6) == 250.0
    assert np.isnan(metres_to_target(curve, "f1", 0.9))
    out = on_metre_grid(curve, "f1", np.array([0.0, 50.0, 100.0, 300.0]))
    np.testing.assert_allclose(out[:3], [0.1, 0.1, 0.5])
    assert np.isnan(out[3])
    comps = pd.DataFrame({"cu": [0.1, 0.5, 0.9], "length_m": [10.0, 10.0, 5.0]})
    assert ore_metres(comps, 0.3) == 15.0


def test_backtest_same_budget_for_every_strategy(prepared_cfg: Config, tmp_path: Path) -> None:
    cfg = prepared_cfg.model_copy(
        update={
            "backtest": prepared_cfg.backtest.model_copy(
                update={
                    "strategies": ["adaptive", "grid", "random"],
                    "random_seeds": [1, 2],
                    "seed_sets": [
                        SeedSet(name="a", method="spread", n_seed=5, seed=1),
                        SeedSet(name="b", method="spread", n_seed=5, seed=2),
                    ],
                    "metre_budget": 300.0,
                    "max_holes": 50,
                }
            )
        }
    )
    results = run_backtest(cfg, tmp_path / "bt")
    runs = results.groupby(["seed_set", "strategy", "rng_seed"])
    assert runs.ngroups == 2 * (1 + 1 + 2)
    for _, r in runs:
        assert r["iteration"].iloc[0] == 0 and r["holes"].iloc[0] == 0
        assert r["cum_metres"].is_monotonic_increasing
        # Each run stops at the first hole that reaches the shared budget.
        assert r["cum_metres"].iloc[-2] < 300.0 <= r["cum_metres"].iloc[-1] or len(r) == 1
    # Same candidate pool within a seed set: identical undrilled count before drilling.
    starts = results[results["iteration"] == 0].groupby("seed_set")["n_undrilled"].nunique()
    assert (starts == 1).all()
    summary = json.loads((tmp_path / "bt" / "summary.json").read_text())
    assert set(summary["strategies"]) == {"adaptive", "grid", "random"}
    assert set(summary["discovery_equivalent"]["vs"]) == {"grid", "random"}
    assert summary["discovery_equivalent"]["label"] == "ESTIMATE"
    assert (tmp_path / "bt" / "backtest.html").exists()


def test_beta_sweep_joins_main_backtest(prepared_cfg: Config, tmp_path: Path) -> None:
    cfg = prepared_cfg.model_copy(
        update={
            "backtest": prepared_cfg.backtest.model_copy(
                update={
                    "strategies": ["adaptive", "random"],
                    "random_seeds": [1],
                    "seed_sets": [SeedSet(name="a", method="spread", n_seed=5, seed=1)],
                    "metre_budget": 300.0,
                    "max_holes": 50,
                    "beta_sweep": [2.0],
                }
            )
        }
    )
    run_backtest(cfg, tmp_path / "bt")
    results = run_beta_sweep(cfg, tmp_path / "bt")
    labels = set(results["strategy"])
    assert labels == {f"adaptive-b{cfg.planner.beta:g}", "adaptive-b2", "random"}
    assert set(results.loc[results["strategy"] == "adaptive-b2", "beta"]) == {2.0}
    summary = json.loads((tmp_path / "bt" / "beta_sweep" / "summary.json").read_text())
    assert "bacc_undrilled" in summary["strategies"]["adaptive-b2"]
