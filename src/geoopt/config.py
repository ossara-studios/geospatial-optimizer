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


class TableSource(_Strict):
    file: str  # relative to data.raw_dir; .csv or .xlsx
    sheet: str | None = None  # Excel sheet name
    columns: dict[str, str]  # canonical name -> raw column name


class DataConfig(_Strict):
    raw_dir: Path
    processed_dir: Path
    # Everything below must be confirmed against the real files (docs/data_schema.md);
    # `geoopt prepare` refuses to run while any of it is null.
    collars: TableSource | None = None
    surveys: TableSource | None = None
    assays: TableSource | None = None
    depth_unit: Literal["m", "ft"] | None = None
    cu_unit: Literal["pct", "ppm"] | None = None
    dip_convention: Literal["negative_down", "positive_down"] | None = None
    # Below-detection assays are stored as -DL; they become DL * this factor.
    below_detection_factor: float = Field(0.5, ge=0, le=1)
    # Local-coordinate origin in source CRS units; null = rounded minimum of collars.
    origin: tuple[float, float, float] | None = None


class CompositeConfig(_Strict):
    length_m: float = Field(gt=0)
    min_coverage: float = Field(ge=0, le=1)


class SplitConfig(_Strict):
    n_seed: int = Field(gt=0)
    method: Literal["earliest", "spread"]


class ModelConfig(_Strict):
    # Model target is log(cu + log_eps), cu in %.
    log_eps: float = Field(gt=0)
    # Ore cutoff, % Cu. Placeholder until the team picks one from the deposit's resource report.
    cutoff_pct: float = Field(gt=0)
    max_train: int = Field(gt=0)
    fit_iters: int = Field(gt=0)
    lr: float = Field(gt=0)
    refit_every: int = Field(ge=0)  # 0 = never re-tune after the seed fit
    cv_folds: int = Field(ge=2)


class Config(_Strict):
    seed: int
    log_level: str = "INFO"
    data: DataConfig
    composite: CompositeConfig
    split: SplitConfig
    model: ModelConfig
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
    for key in ("raw_dir", "processed_dir"):
        raw["data"][key] = (base / raw["data"][key]).resolve()
    return Config.model_validate(raw)
