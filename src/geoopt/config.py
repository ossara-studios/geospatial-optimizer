"""Typed configuration loaded from config/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Timeouts(_Strict):
    connect_s: float = Field(10.0, gt=0)
    check_s: float = Field(15.0, gt=0)
    approval_s: float | None = Field(None, gt=0)


class OpcUaConfig(_Strict):
    endpoint: str
    security: Literal["none"] = "none"
    poll_interval_s: float = Field(0.2, gt=0)
    log_wait_every_s: float = Field(10.0, gt=0)
    timeouts: Timeouts = Timeouts()


class Limits(_Strict):
    max_depth_m: float = Field(gt=0)


class NodeSpec(_Strict):
    id: str | None = None
    node_id: str | None = None
    type: Literal["UInt32", "Int16", "Double"]

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> Self:
        if (self.id is None) == (self.node_id is None):
            raise ValueError("node needs exactly one of `id` or `node_id`")
        return self


class NodeMap(_Strict):
    namespace_uri: str
    nodes: dict[str, NodeSpec]


class Config(_Strict):
    seed: int
    log_level: str = "INFO"
    opcua: OpcUaConfig
    limits: Limits
    nodes: NodeMap


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the main config and the limits/node-map files it references."""
    raw = _read_yaml(path)
    base = path.parent
    raw["limits"] = _read_yaml(base / raw.pop("limits_file"))
    raw["nodes"] = _read_yaml(base / raw.pop("nodes_file"))
    return Config.model_validate(raw)
