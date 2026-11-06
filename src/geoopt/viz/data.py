"""Read-only access to run artifacts for the visualizer. Every reader tolerates partial runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class RunView:
    run_dir: Path
    config: dict
    iterations: list[int]
    lab_dir: Path


def list_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def open_run(run_dir: Path) -> RunView:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    iterations = sorted(
        int(p.name.split("_")[1]) for p in run_dir.glob("iter_*") if (p / "record.json").exists()
    )
    return RunView(run_dir, config, iterations, run_dir / "lab")


def records(run: RunView) -> pd.DataFrame:
    rows = []
    for i in run.iterations:
        try:
            rows.append(json.loads((run.run_dir / f"iter_{i:03d}" / "record.json").read_text()))
        except (OSError, json.JSONDecodeError):
            continue  # being written right now
    return pd.DataFrame(rows)


def grid(run: RunView, iteration: int) -> pd.DataFrame | None:
    path = run.run_dir / f"iter_{iteration:03d}" / "grid.parquet"
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def drilled_composites(run: RunView, hole_idxs: list[int]) -> pd.DataFrame:
    parts = []
    for h in hole_idxs:
        try:
            parts.append(pd.read_parquet(run.lab_dir / f"{h}.parquet"))
        except (OSError, ValueError):
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def planner_tables(run: RunView) -> dict[str, pd.DataFrame]:
    """Seed composites, seed collars and candidate geometry. Planner-visible: no held-out grades."""
    processed = Path(run.config["data"]["processed_dir"])
    out = {}
    for name in ("seed_composites", "candidates", "candidate_points", "collars"):
        try:
            out[name] = pd.read_parquet(processed / f"{name}.parquet")
        except (OSError, ValueError):
            out[name] = pd.DataFrame()
    seed_ids = set(out["seed_composites"].get("hole_id", []))
    collars = out.pop("collars")
    out["seed_collars"] = collars[collars["hole_id"].isin(seed_ids)] if len(collars) else collars
    return out
