"""Composition root: wires prepared data, model, strategy, simulators and a bridge into a run.

This is the only place the held-out truth path is handed to anything, and it goes to LabSim.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geoopt.bridge.base import Bridge
from geoopt.bridge.inprocess import InProcessBridge
from geoopt.config import Config
from geoopt.data.prepare import prepared_paths
from geoopt.loop import DecisionLoop, IterationRecord
from geoopt.model.gp import GPOreModel
from geoopt.planner.strategies import make_strategy
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim

log = logging.getLogger(__name__)


def new_run_dir(cfg: Config, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return cfg.loop.runs_dir / f"{stamp}-{label}"


def build_inprocess_loop(
    cfg: Config,
    strategy_name: str,
    run_dir: Path,
    rng_seed: int | None = None,
    write_grid: bool = True,
) -> tuple[DecisionLoop, LabSim]:
    seed = cfg.seed if rng_seed is None else rng_seed
    paths = prepared_paths(cfg.data.processed_dir)
    seed_comps = pd.read_parquet(paths.seed_composites)
    candidates = pd.read_parquet(paths.candidates)
    points = pd.read_parquet(paths.candidate_points)
    collars = pd.read_parquet(paths.collars)
    seed_xy = collars[collars["hole_id"].isin(seed_comps["hole_id"].unique())][["x", "y"]]

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json") | {"strategy": strategy_name, "rng_seed": seed})
    )
    model = GPOreModel(cfg.model, seed)
    model.fit(seed_comps[["x", "y", "z"]].to_numpy(), seed_comps["cu"].to_numpy())

    lab_dir = run_dir / "lab"
    lab = LabSim(paths.heldout_composites, lab_dir)
    bridge: Bridge = InProcessBridge(cfg.limits, RigSim(candidates, seed), lab)
    loop = DecisionLoop(
        cfg=cfg.loop,
        strategy=make_strategy(strategy_name, cfg.planner, candidates, seed),
        bridge=bridge,
        model=model,
        candidates=candidates,
        points=points,
        seed_collars_xy=seed_xy.to_numpy(dtype=float),
        run_dir=run_dir,
        lab_dir=lab_dir,
        write_grid=write_grid,
    )
    return loop, lab


async def run_inprocess(
    cfg: Config, strategy_name: str, run_dir: Path | None = None
) -> list[IterationRecord]:
    run_dir = run_dir or new_run_dir(cfg, strategy_name)
    loop, _ = build_inprocess_loop(cfg, strategy_name, run_dir)
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
