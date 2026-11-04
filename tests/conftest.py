from __future__ import annotations

import socket
from pathlib import Path

import pytest

from geoopt.config import Config, load_config

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
