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
    drill_s: float | None = Field(None, gt=0)  # wait for Done.ReqId; null = indefinitely


class OpcUaConfig(_Strict):
    endpoint: str
    security: Literal["none"] = "none"
    poll_interval_s: float = Field(0.2, gt=0)
    log_wait_every_s: float = Field(10.0, gt=0)
    timeouts: Timeouts = Timeouts()


class Bounds(_Strict):
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class RectZone(_Strict):
    kind: Literal["rect"]
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class CircleZone(_Strict):
    kind: Literal["circle"]
    name: str
    x: float
    y: float
    radius: float = Field(gt=0)


class Limits(_Strict):
    """Mirrored by the EAE limit-check FB. Coordinates are local metres (see meta.json)."""

    max_depth_m: float = Field(gt=0)
    site_bounds: Bounds | None = None
    no_go_zones: list[RectZone | CircleZone] = []
    dip_range: tuple[float, float] = (-90.0, 0.0)  # degrees, negative = down
    azimuth_range: tuple[float, float] = (0.0, 360.0)


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
    seed: int | None = None  # RNG for the spread split; null = top-level seed


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


class PlannerConfig(_Strict):
    beta: float = Field(ge=0)  # weight on exploration (uncertainty) vs expected ore
    # z-score both terms across candidates each iteration so beta is unit-free.
    standardize: bool = True
    min_spacing_m: float = Field(ge=0)
    spacing_weight: float = Field(ge=0)
    grid_spacing_m: float = Field(gt=0)  # GridStrategy node spacing


class RigConfig(_Strict):
    metres_per_second: float = Field(gt=0)  # demo drilling speed, not a real rate
    telemetry_period_s: float = Field(gt=0)


class LoopConfig(_Strict):
    runs_dir: Path
    max_holes: int = Field(gt=0)
    max_metres: float | None = Field(None, gt=0)
    # Fixed 3D evaluation grid (nx, ny, nz) written each iteration for the visualizer.
    eval_grid: tuple[int, int, int] = (30, 30, 12)
    # Lab drop folder shared by the rig simulator and the planner in OPC UA mode.
    live_lab_dir: Path


class SeedSet(_Strict):
    name: str
    method: Literal["earliest", "spread"]
    n_seed: int = Field(gt=0)
    seed: int = 0


class BacktestConfig(_Strict):
    out_dir: Path
    strategies: list[Literal["adaptive", "grid", "historical", "random"]]
    random_seeds: list[int]
    seed_sets: list[SeedSet]
    metre_budget: float = Field(gt=0)
    max_holes: int = Field(gt=0)
    ore_threshold: float = Field(0.5, gt=0, lt=1)  # classify ore when P(ore) > this
    f1_target: float = Field(gt=0, le=1)  # "metres to target" uses this delineation F1
    # Near-data delineation scores undrilled composites within this distance of any assayed one.
    near_radius_m: float = Field(50.0, gt=0)
    beta_sweep: list[float] = []  # extra adaptive betas for `geoopt backtest --beta-sweep`


class EnvFactors(_Strict):
    """ESTIMATES used to translate metres/holes avoided into environmental terms."""

    source_note: str
    water_l_per_m: float = Field(ge=0)
    diesel_l_per_m: float = Field(ge=0)
    pad_area_m2_per_hole: float = Field(ge=0)


class Config(_Strict):
    seed: int
    log_level: str = "INFO"
    data: DataConfig
    loop: LoopConfig
    rig: RigConfig
    composite: CompositeConfig
    split: SplitConfig
    model: ModelConfig
    planner: PlannerConfig
    backtest: BacktestConfig
    env: EnvFactors
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
    raw["env"] = _read_yaml(base / raw.pop("env_file"))
    raw["backtest"]["out_dir"] = (base / raw["backtest"]["out_dir"]).resolve()
    for key in ("raw_dir", "processed_dir"):
        raw["data"][key] = (base / raw["data"][key]).resolve()
    for key in ("runs_dir", "live_lab_dir"):
        raw["loop"][key] = (base / raw["loop"][key]).resolve()
    return Config.model_validate(raw)
