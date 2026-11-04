"""Decision loop end to end with the in-process bridge, plus leakage guards."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import geoopt
from geoopt.config import Config, RectZone
from geoopt.data.prepare import prepare
from geoopt.run import build_inprocess_loop, summarize

PACKAGE = Path(geoopt.__file__).parent
# Modules on the planner side of the wall: they must never reach held-out data.
PLANNER_SIDE = [
    *sorted((PACKAGE / "planner").glob("*.py")),
    *sorted((PACKAGE / "model").glob("*.py")),
    PACKAGE / "loop.py",
    PACKAGE / "bridge" / "base.py",
]


@pytest.mark.parametrize("path", PLANNER_SIDE, ids=lambda p: p.name)
def test_planner_side_cannot_reach_truth(path: Path) -> None:
    src = path.read_text()
    assert not re.search(r"geoopt\.sim|from geoopt import sim", src), "imports the simulator"
    assert not re.search(r"truth|heldout|held_out", src, re.IGNORECASE), "references truth"


@pytest.mark.parametrize("strategy", ["adaptive", "random", "grid"])
async def test_loop_runs_end_to_end(prepared_cfg: Config, tmp_path: Path, strategy: str) -> None:
    prepare(prepared_cfg)
    run_dir = tmp_path / "runs" / strategy
    loop, _ = build_inprocess_loop(prepared_cfg, strategy, run_dir)
    n_seed_train = loop._model.n_train
    records = await loop.run()
    summary = summarize(records)

    assert summary["holes_drilled"] == prepared_cfg.loop.max_holes
    drilled = [r for r in records if r.metres_drilled > 0]
    assert len({r.hole_idx for r in drilled}) == len(drilled)  # never drilled twice
    assert [r.req_id for r in records] == list(range(1, len(records) + 1))
    # The model only ever saw seed data plus what the lab released.
    lab_files = sorted((run_dir / "lab").glob("*.parquet"))
    assert {int(p.stem) for p in lab_files} == {r.hole_idx for r in drilled}
    released = sum(len(pd.read_parquet(p)) for p in lab_files)
    assert loop._model.n_train == n_seed_train + released
    # Artifacts per iteration.
    last = run_dir / f"iter_{len(records):03d}"
    assert json.loads((last / "record.json").read_text())["req_id"] == len(records)
    assert len(pd.read_parquet(last / "grid.parquet")) == 6 * 6 * 4
    assert (run_dir / "config.yaml").exists()


async def test_clipped_depth_limits_released_assays(prepared_cfg: Config, tmp_path: Path) -> None:
    cfg = prepared_cfg.model_copy(
        update={"limits": prepared_cfg.limits.model_copy(update={"max_depth_m": 60.0})}
    )
    prepare(cfg)
    loop, _ = build_inprocess_loop(cfg, "random", tmp_path / "run", write_grid=False)
    records = await loop.run()
    assert all(r.check_status == "CLIPPED" for r in records)
    assert all(r.metres_drilled <= 60.0 for r in records)
    for p in (tmp_path / "run" / "lab").glob("*.parquet"):
        assert pd.read_parquet(p)["to_m"].max() <= 60.0


async def test_rejected_holes_are_excluded(prepared_cfg: Config, tmp_path: Path) -> None:
    prepare(prepared_cfg)
    meta = json.loads((prepared_cfg.data.processed_dir / "meta.json").read_text())
    assert meta["n_candidates"] > 0
    # A no-go zone covering everything: every request is rejected, nothing is drilled.
    zone = RectZone(kind="rect", name="all", x_min=-1e9, x_max=1e9, y_min=-1e9, y_max=1e9)
    cfg = prepared_cfg.model_copy(
        update={"limits": prepared_cfg.limits.model_copy(update={"no_go_zones": [zone]})}
    )
    loop, _ = build_inprocess_loop(cfg, "random", tmp_path / "run", write_grid=False)
    records = await loop.run()
    assert len(records) == meta["n_candidates"]
    assert all(r.check_status == "REJECTED" and r.metres_drilled == 0 for r in records)
