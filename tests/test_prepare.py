"""End-to-end `prepare` on a small synthetic drillhole database written as CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoopt.config import Config, DataConfig, TableSource
from geoopt.data.load import DataConfigIncomplete, clean_cu, load_tables
from geoopt.data.prepare import prepare

N_HOLES = 30


def _write_raw(raw_dir: Path) -> None:
    rng = np.random.default_rng(3)
    ids = [f"CC-{i:03d}" for i in range(N_HOLES)]
    depth_ft = rng.uniform(300, 900, N_HOLES)
    pd.DataFrame(
        {
            "HOLEID": ids,
            "EAST": 530_000 + rng.uniform(0, 2000, N_HOLES),
            "NORTH": 3_620_000 + rng.uniform(0, 2000, N_HOLES),
            "RL": rng.uniform(900, 1100, N_HOLES),
            "TD": depth_ft,
        }
    ).to_csv(raw_dir / "collars.csv", index=False)
    pd.DataFrame(
        {"HOLEID": ids, "AT": 0.0, "AZ": rng.uniform(0, 360, N_HOLES), "DIP": 70.0}
    ).to_csv(raw_dir / "surveys.csv", index=False)
    rows = []
    for hid, td in zip(ids, depth_ft, strict=True):
        edges = np.arange(0.0, td, 10.0)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            rows.append((hid, a, b, rng.choice([rng.lognormal(-1.5, 1.0), -0.001, np.nan])))
    pd.DataFrame(rows, columns=["HOLEID", "FROM", "TO", "CU_PCT"]).to_csv(
        raw_dir / "assays.csv", index=False
    )


def _data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        collars=TableSource(
            file="collars.csv",
            columns={
                "hole_id": "HOLEID",
                "x": "EAST",
                "y": "NORTH",
                "z": "RL",
                "total_depth": "TD",
            },
        ),
        surveys=TableSource(
            file="surveys.csv",
            columns={"hole_id": "HOLEID", "depth": "AT", "azimuth": "AZ", "dip": "DIP"},
        ),
        assays=TableSource(
            file="assays.csv",
            columns={"hole_id": "HOLEID", "from_m": "FROM", "to_m": "TO", "cu": "CU_PCT"},
        ),
        depth_unit="ft",
        cu_unit="pct",
        dip_convention="positive_down",
    )


@pytest.fixture
def prepared_cfg(cfg: Config, tmp_path: Path) -> Config:
    (tmp_path / "raw").mkdir()
    _write_raw(tmp_path / "raw")
    return cfg.model_copy(
        update={
            "data": _data_cfg(tmp_path),
            "split": cfg.split.model_copy(update={"n_seed": 5}),
        }
    )


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
