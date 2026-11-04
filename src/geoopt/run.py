"""Composition root: wires prepared data, model, strategy, simulators and a bridge into a run.

This is the only place the held-out truth path is handed to anything, and it goes to LabSim.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geoopt.bridge.base import Bridge
from geoopt.bridge.inprocess import InProcessBridge
from geoopt.bridge.opcua import OpcUaBridge
from geoopt.config import Config
from geoopt.data.prepare import prepared_paths
from geoopt.loop import DecisionLoop, IterationRecord
from geoopt.model.gp import GPOreModel
from geoopt.planner.strategies import make_strategy
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim
from geoopt.sim.rig_client import serve_rig

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
    )


def build_inprocess_loop(
    cfg: Config,
    strategy_name: str,
    run_dir: Path,
    rng_seed: int | None = None,
    write_grid: bool = True,
) -> tuple[DecisionLoop, LabSim]:
    seed = cfg.seed if rng_seed is None else rng_seed
    inputs = load_planner_inputs(cfg)
    lab_dir = run_dir / "lab"
    lab = LabSim(prepared_paths(cfg.data.processed_dir).heldout_composites, lab_dir)
    bridge = InProcessBridge(cfg.limits, RigSim(inputs.candidates, seed), lab)
    loop = build_loop(cfg, inputs, strategy_name, bridge, run_dir, lab_dir, seed, write_grid)
    return loop, lab


async def run_inprocess(
    cfg: Config, strategy_name: str, run_dir: Path | None = None
) -> list[IterationRecord]:
    run_dir = run_dir or new_run_dir(cfg, strategy_name)
    loop, _ = build_inprocess_loop(cfg, strategy_name, run_dir)
    records = await loop.run()
    log.info("run written to %s", run_dir)
    return records


async def run_opcua(
    cfg: Config, strategy_name: str, run_dir: Path | None = None
) -> list[IterationRecord]:
    """Planner side of a live run. EAE (or the emulator) and `geoopt rig-sim` run separately."""
    run_dir = run_dir or new_run_dir(cfg, f"{strategy_name}-opcua")
    inputs = load_planner_inputs(cfg)
    async with OpcUaBridge.connect(cfg) as bridge:
        loop = build_loop(cfg, inputs, strategy_name, bridge, run_dir, cfg.loop.live_lab_dir)
        records = await loop.run()
    log.info("run written to %s", run_dir)
    return records


async def run_rig_sim(cfg: Config) -> None:
    """Rig + lab simulator for live runs. Owns the held-out truth; clears old lab drops."""
    paths = prepared_paths(cfg.data.processed_dir)
    shutil.rmtree(cfg.loop.live_lab_dir, ignore_errors=True)
    lab = LabSim(paths.heldout_composites, cfg.loop.live_lab_dir)
    rig = RigSim(pd.read_parquet(paths.candidates), cfg.seed)
    await serve_rig(cfg, rig, lab)


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
