from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoopt.config import ModelConfig
from geoopt.model.gp import GPOreModel
from geoopt.model.transforms import CoordScaler, from_log_grade, to_log_grade
from geoopt.model.validate import cross_validate

CFG = ModelConfig(
    log_eps=0.001,
    cutoff_pct=0.3,
    max_train=2000,
    fit_iters=80,
    lr=0.1,
    refit_every=0,
    cv_folds=4,
)


def _field(xyz: np.ndarray) -> np.ndarray:
    """Smooth anisotropic log-grade field: long in x, short in z."""
    x, y, z = xyz.T
    return -1.5 + 1.2 * np.sin(x / 300.0) + 0.8 * np.cos(y / 150.0) + 0.5 * np.sin(z / 60.0)


def _holes(n_holes: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for h in range(n_holes):
        cx, cy = rng.uniform(0, 1000, 2)
        depth = np.arange(5.0, 200.0, 10.0)
        for d in depth:
            rows.append((f"H{h:02d}", cx, cy, -d))
    df = pd.DataFrame(rows, columns=["hole_id", "x", "y", "z"])
    log_cu = _field(df[["x", "y", "z"]].to_numpy()) + rng.normal(0, 0.15, len(df))
    df["cu"] = from_log_grade(log_cu, CFG.log_eps).clip(min=0)
    return df


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _holes(30, np.random.default_rng(0))


def test_transforms_round_trip() -> None:
    xyz = np.random.default_rng(1).normal(size=(50, 3)) * [100, 50, 10] + [1000, 2000, 30]
    scaler = CoordScaler.fit(xyz)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(xyz)), xyz)
    cu = np.array([0.0, 0.05, 1.2])
    np.testing.assert_allclose(from_log_grade(to_log_grade(cu, 0.001), 0.001), cu, atol=1e-12)
    with pytest.raises(ValueError):
        to_log_grade(np.array([np.nan]), 0.001)


def test_fit_beats_mean_and_is_calibrated(data: pd.DataFrame) -> None:
    report = cross_validate(data, CFG, seed=0)
    assert report.rmse_log < 0.6 * report.rmse_log_baseline
    assert 0.55 < report.coverage_1sd < 0.85
    assert 0.88 < report.coverage_2sd <= 1.0


def test_condition_shrinks_uncertainty_and_keeps_hyperparameters(data: pd.DataFrame) -> None:
    seed_holes = data["hole_id"].unique()[:10]
    seed = data[data["hole_id"].isin(seed_holes)]
    new = data[data["hole_id"] == data["hole_id"].unique()[20]]
    model = GPOreModel(CFG, seed=0)
    model.fit(seed[["x", "y", "z"]].to_numpy(), seed["cu"].to_numpy())
    hyper = model.hyperparameters()
    xyz_new = new[["x", "y", "z"]].to_numpy()
    _, std_before = model.predict(xyz_new)
    model.condition(xyz_new, new["cu"].to_numpy())
    _, std_after = model.predict(xyz_new)
    assert (std_after < std_before).all()
    assert model.hyperparameters() == hyper
    assert model.n_train == len(seed) + len(new)


def test_prob_ore_is_monotone_in_cutoff(data: pd.DataFrame) -> None:
    model = GPOreModel(CFG, seed=0)
    model.fit(data[["x", "y", "z"]].to_numpy(), data["cu"].to_numpy())
    pts = data[["x", "y", "z"]].to_numpy()[:50]
    low, high = model.prob_ore(pts, 0.1), model.prob_ore(pts, 1.0)
    assert ((low >= high) & (low <= 1) & (high >= 0)).all()


def test_refit_every_retunes(data: pd.DataFrame) -> None:
    cfg = CFG.model_copy(update={"refit_every": 1})
    holes = data["hole_id"].unique()
    model = GPOreModel(cfg, seed=0)
    first = data[data["hole_id"].isin(holes[:5])]
    model.fit(first[["x", "y", "z"]].to_numpy(), first["cu"].to_numpy())
    hyper = model.hyperparameters()
    more = data[data["hole_id"].isin(holes[5:15])]
    model.condition(more[["x", "y", "z"]].to_numpy(), more["cu"].to_numpy())
    assert model.hyperparameters() != hyper


def test_variance_reduction_matches_conditioning() -> None:
    """Closed-form reduction equals the drop in latent variance after actually conditioning."""
    import copy

    rng = np.random.default_rng(3)
    df = _holes(8, rng)
    cfg = CFG.model_copy(update={"device": "cpu", "dtype": "float64", "fit_iters": 30})
    model = GPOreModel(cfg, seed=0)
    model.fit(df[["x", "y", "z"]].to_numpy(), df["cu"].to_numpy())
    targets = rng.uniform([0, 0, -200], [1000, 1000, 0], (40, 3))
    w = rng.uniform(0.5, 1.5, len(targets))
    groups = {7: rng.uniform([0, 0, -200], [1000, 1000, 0], (5, 3))}
    groups[2] = rng.uniform([0, 0, -200], [1000, 1000, 0], (3, 3))
    pts = np.vstack(list(groups.values()))
    gid = np.concatenate([np.full(len(v), k) for k, v in groups.items()])
    ids, red = model.variance_reduction(targets, w, pts, gid)
    _, before = model.predict(targets)
    for g, r in zip(ids, red, strict=True):
        after_model = copy.deepcopy(model)
        after_model.condition(groups[g], np.full(len(groups[g]), 0.5))
        _, after = after_model.predict(targets)
        assert r == pytest.approx(float(np.sum(w * (before**2 - after**2))), rel=1e-6)
