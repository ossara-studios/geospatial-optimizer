from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoopt.config import Config, DataConfig, TableSource, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg() -> Config:
    return load_config(REPO_ROOT / "config" / "default.yaml")


@pytest.fixture
def local_cfg(cfg: Config) -> Config:
    """Config pointing at a free localhost port, with short timeouts for tests."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    opcua = cfg.opcua.model_copy(
        update={
            "endpoint": f"opc.tcp://127.0.0.1:{port}/geoopt/",
            "poll_interval_s": 0.05,
            "timeouts": cfg.opcua.timeouts.model_copy(update={"check_s": 3.0, "approval_s": 3.0}),
        }
    )
    return cfg.model_copy(update={"opcua": opcua})


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
    """Config over a small synthetic drillhole database (feet, positive-down dips)."""
    (tmp_path / "raw").mkdir()
    _write_raw(tmp_path / "raw")
    return cfg.model_copy(
        update={
            "data": _data_cfg(tmp_path),
            "split": cfg.split.model_copy(update={"n_seed": 5}),
            "loop": cfg.loop.model_copy(
                update={"runs_dir": tmp_path / "runs", "max_holes": 4, "eval_grid": (6, 6, 4)}
            ),
            "model": cfg.model.model_copy(update={"fit_iters": 40}),
        }
    )
