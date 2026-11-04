"""Visualizer readers and figures on a real (synthetic-data) in-process run."""

from __future__ import annotations

import asyncio
from pathlib import Path

from geoopt.config import Config
from geoopt.data.prepare import prepare
from geoopt.run import run_inprocess
from geoopt.viz import data, figures


def test_viz_reads_run_and_builds_figures(prepared_cfg: Config, tmp_path: Path) -> None:
    prepare(prepared_cfg)
    run_dir = tmp_path / "run"
    asyncio.run(run_inprocess(prepared_cfg, "adaptive", run_dir))

    assert data.list_runs(tmp_path) == [run_dir]
    run = data.open_run(run_dir)
    recs = data.records(run)
    assert len(recs) == len(run.iterations) > 0
    tables = data.planner_tables(run)
    assert len(tables["seed_collars"]) == prepared_cfg.split.n_seed
    grid = data.grid(run, run.iterations[-1])
    assert grid is not None

    fig = figures.plan_map(
        tables["candidates"], tables["seed_collars"], recs, run.config["limits"], grid
    )
    assert any(t.name == "Drilled by planner" for t in fig.data)
    drilled = recs.loc[recs["metres_drilled"] > 0, "hole_idx"].astype(int).tolist()
    lab = data.drilled_composites(run, drilled)
    assert set(lab["hole_idx"]) <= set(drilled)
    figures.section(grid, lab, float(grid["x"].mean()), run.config["model"]["cutoff_pct"])
    figures.progress(recs)
