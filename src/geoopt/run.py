"""Composition root: wires prepared data, model, strategy and simulators into a run.

This is the only place the held-out truth path is handed to anything, and it goes to LabSim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geoopt.bridge.base import Bridge
from geoopt.bridge.inprocess import InProcessBridge
from geoopt.config import Config
from geoopt.data.prepare import prepared_paths
from geoopt.kpi import DiscoveryTracker, kpi_geometry, load_baseline
from geoopt.loop import DecisionLoop, IterationRecord
from geoopt.model.gp import GPOreModel
from geoopt.planner.strategies import make_strategy
from geoopt.progress import Progress
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerInputs:
    """Everything the planner side may read (no held-out grades)."""

    seed_composites: pd.DataFrame
    candidates: pd.DataFrame
    points: pd.DataFrame
    seed_collars_xy: np.ndarray


def load_planner_inputs(cfg: Config) -> PlannerInputs:
    paths = prepared_paths(cfg.data.processed_dir)
    seed = pd.read_parquet(paths.seed_composites)
    collars = pd.read_parquet(paths.collars)
    seed_xy = collars[collars["hole_id"].isin(seed["hole_id"].unique())][["x", "y"]]
    return PlannerInputs(
        seed_composites=seed,
        candidates=pd.read_parquet(paths.candidates),
        points=pd.read_parquet(paths.candidate_points),
        seed_collars_xy=seed_xy.to_numpy(dtype=float),
    )


def new_run_dir(cfg: Config, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return cfg.loop.runs_dir / f"{stamp}-{label}"


def build_loop(
    cfg: Config,
    inputs: PlannerInputs,
    strategy_name: str,
    bridge: Bridge,
    run_dir: Path,
    lab_dir: Path,
    rng_seed: int | None = None,
    write_grid: bool = True,
    with_kpis: bool = False,
    progress: Progress | None = None,
) -> DecisionLoop:
    seed = cfg.seed if rng_seed is None else rng_seed
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json") | {"strategy": strategy_name, "rng_seed": seed})
    )
    model = GPOreModel(cfg.model, seed)
    sc = inputs.seed_composites
    model.fit(sc[["x", "y", "z"]].to_numpy(), sc["cu"].to_numpy())
    return DecisionLoop(
        cfg=cfg.loop,
        strategy=make_strategy(strategy_name, cfg.planner, inputs.candidates, seed),
        bridge=bridge,
        model=model,
        candidates=inputs.candidates,
        points=inputs.points,
        seed_collars_xy=inputs.seed_collars_xy,
        run_dir=run_dir,
        lab_dir=lab_dir,
        write_grid=write_grid,
        kpi=discovery_tracker(cfg, inputs) if with_kpis else None,
        progress=progress,
    )


def discovery_tracker(cfg: Config, inputs: PlannerInputs) -> DiscoveryTracker:
    baseline = load_baseline(cfg.kpi.baseline_file)
    if baseline is None:
        log.warning(
            "no near-ore baseline at %s: run `geoopt baseline` for cost saved",
            cfg.kpi.baseline_file,
        )
    return DiscoveryTracker(
        cfg.kpi,
        cfg.model.cutoff_pct,
        cfg.backtest.block_size_m,
        kpi_geometry(inputs.points, inputs.seed_composites, cfg.limits.max_depth_m),
        inputs.seed_composites,
        baseline,
    )


def build_inprocess_loop(
    cfg: Config,
    strategy_name: str,
    run_dir: Path,
    rng_seed: int | None = None,
    write_grid: bool = True,
    with_kpis: bool = False,
    progress: Progress | None = None,
) -> tuple[DecisionLoop, LabSim]:
    seed = cfg.seed if rng_seed is None else rng_seed
    inputs = load_planner_inputs(cfg)
    lab_dir = run_dir / "lab"
    lab = LabSim(prepared_paths(cfg.data.processed_dir).heldout_composites, lab_dir)
    rig = RigSim(inputs.candidates)
    bridge = InProcessBridge(cfg.limits, rig, lab, progress, cfg.rig.metres_per_second)
    loop = build_loop(
        cfg, inputs, strategy_name, bridge, run_dir, lab_dir, seed, write_grid, with_kpis, progress
    )
    return loop, lab


async def run_inprocess(
    cfg: Config,
    strategy_name: str,
    run_dir: Path | None = None,
    progress: Progress | None = None,
) -> list[IterationRecord]:
    """One run, instant by default; with `progress` it runs at demo pace for the live view."""
    run_dir = run_dir or new_run_dir(cfg, strategy_name)
    loop, _ = build_inprocess_loop(cfg, strategy_name, run_dir, with_kpis=True, progress=progress)
    records = await loop.run()
    log.info("run written to %s", run_dir)
    return records


def summarize(records: list[IterationRecord]) -> dict[str, float]:
    drilled = [r for r in records if r.metres_drilled > 0]
    return {
        "iterations": len(records),
        "holes_drilled": len(drilled),
        "rejected": sum(r.check_status == "REJECTED" for r in records),
        "metres": float(np.sum([r.metres_drilled for r in drilled])),
        "mean_condition_s": float(np.mean([r.timings_s.get("condition", 0) for r in drilled]))
        if drilled
        else 0.0,
        "mean_recommend_s": float(np.mean([r.timings_s["recommend"] for r in records]))
        if records
        else 0.0,
    }
