"""Typed configuration loaded from config/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    """Drilling limits. Coordinates are local metres (see meta.json)."""

    max_depth_m: float = Field(gt=0)
    site_bounds: Bounds | None = None
    no_go_zones: list[RectZone | CircleZone] = []
    dip_range: tuple[float, float] = (-90.0, 0.0)  # degrees, negative = down
    azimuth_range: tuple[float, float] = (0.0, 360.0)


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
    # Ore cutoff, % Cu. Rex Minerals report the Hillside Mineral Resource (Dec 2022) at 0.2 % Cu.
    cutoff_pct: float = Field(gt=0)
    max_train: int = Field(gt=0)
    fit_iters: int = Field(gt=0)
    lr: float = Field(gt=0)
    refit_every: int = Field(ge=0)  # 0 = never re-tune after the seed fit
    cv_folds: int = Field(ge=2)
    # "auto" picks CUDA when torch sees a GPU, else CPU. dtype "auto" is float32 on CUDA
    # (consumer GPUs run float64 at ~1/64 speed) and float64 on CPU, the reference.
    device: Literal["cpu", "cuda", "auto"] = "auto"
    dtype: Literal["float64", "float32", "auto"] = "auto"
    # Frozen seed fit: the tuned hyperparameters are saved here, keyed by the training data and
    # these settings, and loaded instead of re-tuning. Identical results, instant start.
    fit_cache_dir: Path | None = None


class PlannerConfig(_Strict):
    beta: float = Field(ge=0)  # weight on exploration (uncertainty) vs expected ore
    # Weight on boundary uncertainty: latent P(ore) * (1 - P(ore)), highest at ore/waste edges.
    gamma: float = Field(0.0, ge=0)
    # Weight on targeted variance reduction: how much drilling the hole would shrink the
    # latent variance of blocks near the ore/waste boundary across the deposit (tIMSE).
    delta: float = Field(0.0, ge=0)
    ore_weight: float = Field(1.0, ge=0)  # weight on expected ore (0 = pure delineation)
    delin_pool: int = Field(20_000, gt=0)  # fixed random subset of blocks the term considers
    delin_targets: int = Field(2_000, gt=0)  # blocks sampled per iteration, by boundary weight
    delin_point_stride: int = Field(2, gt=0)  # use every k-th sample point along each hole
    block_size_m: tuple[float, float, float] = (25.0, 25.0, 10.0)
    block_radius_m: float = Field(50.0, gt=0)  # blocks within this of a candidate sample
    # z-score both terms across candidates each iteration so beta is unit-free.
    standardize: bool = True
    min_spacing_m: float = Field(ge=0)
    spacing_weight: float = Field(ge=0)
    grid_spacing_m: float = Field(gt=0)  # GridStrategy node spacing


class RigConfig(_Strict):
    metres_per_second: float = Field(gt=0)  # demo drilling speed, not a real rate


class LoopConfig(_Strict):
    runs_dir: Path
    max_holes: int = Field(gt=0)
    max_metres: float | None = Field(None, gt=0)
    # Fixed 3D evaluation grid (nx, ny, nz) written each iteration for the visualizer.
    eval_grid: tuple[int, int, int] = (30, 30, 12)
    # Assays reach the model only after this many more holes are drilled (lab turnaround).
    assay_lag_holes: int = Field(0, ge=0)


class AdaptiveVariant(_Strict):
    beta: float = Field(ge=0)
    gamma: float = Field(0.0, ge=0)
    delta: float = Field(0.0, ge=0)
    ore_weight: float = Field(1.0, ge=0)

    @property
    def label(self) -> str:
        return (
            "adaptive"
            + (f"-o{self.ore_weight:g}" if self.ore_weight != 1 else "")
            + f"-b{self.beta:g}"
            + (f"-g{self.gamma:g}" if self.gamma else "")
            + (f"-d{self.delta:g}" if self.delta else "")
        )


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
    # Summary checkpoints in cumulative metres; empty = 25 %, 50 % and 100 % of the budget.
    report_at_m: list[float] = []
    # Discovery-equivalent savings match each baseline's ore at this many metres (null = budget).
    discovery_at_m: float | None = Field(None, gt=0)
    max_holes: int = Field(gt=0)
    ore_threshold: float = Field(0.5, gt=0, lt=1)  # classify ore when P(ore) > this
    f1_target: float = Field(gt=0, le=1)  # "metres to target" uses this delineation F1
    # Near-data delineation scores undrilled composites within this distance of any assayed one.
    near_radius_m: float = Field(50.0, gt=0)
    # Block-model delineation: each strategy's block estimate vs a reference model fitted on
    # every hole (what the final resource model would be built from). Grade-above-cutoff rule.
    block_size_m: tuple[float, float, float] = (25.0, 25.0, 10.0)
    block_radius_m: float = Field(50.0, gt=0)  # blocks within this of any assayed composite
    block_every_m: float = Field(2_500.0, gt=0)  # score blocks each time this many metres pass
    reference_max_train: int = Field(6_000, gt=0)
    block_f1_targets: list[float] = [0.7, 0.8, 0.9]  # "metres to block F1" reported for each
    beta_sweep: list[float] = []  # extra adaptive betas for `geoopt backtest --beta-sweep`
    boundary_sweep: list[AdaptiveVariant] = []  # for `geoopt backtest --boundary-sweep`
    delineation_sweep: list[AdaptiveVariant] = []  # for `geoopt backtest --delineation-sweep`
    # `geoopt backtest --assay-delay-sweep`: adaptive at assay_delay_beta, each lag in holes.
    assay_delay_sweep: list[int] = []
    assay_delay_beta: float = Field(1.0, ge=0)


class EnvFactors(_Strict):
    """ESTIMATES used to translate metres/holes avoided into environmental terms."""

    source_note: str
    water_l_per_m: float = Field(ge=0)
    diesel_l_per_m: float = Field(ge=0)
    pad_area_m2_per_hole: float = Field(ge=0)


class KpiConfig(_Strict):
    """Live discovery KPIs (geoopt.kpi). Density and drilling cost are stated ASSUMPTIONS."""

    radius_m: float = Field(25.0, gt=0)  # a block is reached by ore composites within this
    block_radius_m: float = Field(50.0, gt=0)  # blocks counted within this of any geometry
    density_t_m3: float = Field(2.8, gt=0)
    drill_cost_aud_per_m: float = Field(373.70, gt=0)
    near_ore_m: float = Field(50.0, gt=0)  # near-ore baseline rule
    baseline_budget_m: float = Field(60_000.0, gt=0)  # long enough to match the planner
    baseline_file: Path | None = None


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
    kpi: KpiConfig = KpiConfig()
    limits: Limits


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load the main config and the limits/env files it references."""
    raw = _read_yaml(path)
    base = path.parent
    raw["limits"] = _read_yaml(base / raw.pop("limits_file"))
    raw["env"] = _read_yaml(base / raw.pop("env_file"))
    raw["backtest"]["out_dir"] = (base / raw["backtest"]["out_dir"]).resolve()
    for key in ("raw_dir", "processed_dir"):
        raw["data"][key] = (base / raw["data"][key]).resolve()
    raw["loop"]["runs_dir"] = (base / raw["loop"]["runs_dir"]).resolve()
    if raw.get("kpi", {}).get("baseline_file"):
        raw["kpi"]["baseline_file"] = (base / raw["kpi"]["baseline_file"]).resolve()
    if raw["model"].get("fit_cache_dir"):
        raw["model"]["fit_cache_dir"] = (base / raw["model"]["fit_cache_dir"]).resolve()
    return Config.model_validate(raw)
