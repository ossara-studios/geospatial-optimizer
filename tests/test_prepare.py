"""End-to-end `prepare` on a small synthetic drillhole database written as CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import N_HOLES

from geoopt.config import Config
from geoopt.data.load import DataConfigIncomplete, clean_cu, load_tables
from geoopt.data.prepare import prepare


def test_prepare_writes_tables_and_keeps_truth_separate(prepared_cfg: Config) -> None:
    paths = prepare(prepared_cfg)
    seed = pd.read_parquet(paths.seed_composites)
    cand = pd.read_parquet(paths.candidates)
    points = pd.read_parquet(paths.candidate_points)
    truth = pd.read_parquet(paths.heldout_composites)

    assert seed["hole_id"].nunique() == 5
    assert len(cand) == N_HOLES - 5
    assert not set(seed["hole_id"]) & set(cand["hole_id"])
    assert "cu" not in cand.columns and "cu" not in points.columns
    assert set(truth["hole_id"]) == set(cand["hole_id"])
    assert paths.heldout_composites.parent != paths.candidates.parent
    # Feet -> metres and positive-down dips -> negative-down, local coordinates.
    assert cand["depth"].max() < 900 * 0.3048 + 1e-6
    assert (cand["dip"] < 0).all()
    assert seed["z"].max() < 300  # local metres, not raw RL
    assert seed["cu"].min() >= 0


def test_prepare_refuses_unconfirmed_config(cfg: Config) -> None:
    with pytest.raises(DataConfigIncomplete, match="collars"):
        load_tables(cfg.data)


def test_clean_cu_handles_below_detection_and_blanks() -> None:
    out = clean_cu(pd.Series([0.5, -0.002, np.nan, 2500.0]), "pct", 0.5)
    np.testing.assert_allclose(out[:2], [0.5, 0.001])
    assert np.isnan(out[2])
    assert clean_cu(pd.Series([2500.0]), "ppm", 0.5).iloc[0] == pytest.approx(0.25)
