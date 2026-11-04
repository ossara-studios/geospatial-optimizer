"""`geoopt prepare`: raw tables -> processed parquet files for the planner and the simulator.

Held-out grades go to a separate `truth/` directory, which only the rig/lab simulator reads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geoopt.config import Config
from geoopt.data.composite import composite
from geoopt.data.desurvey import desurvey
from geoopt.data.load import DrillholeTables, load_tables
from geoopt.data.split import build_candidates, choose_seed_holes

log = logging.getLogger(__name__)

TRUTH_DIR = "truth"


@dataclass(frozen=True)
class PreparedPaths:
    seed_composites: Path
    candidates: Path
    candidate_points: Path
    collars: Path
    meta: Path
    heldout_composites: Path  # ground truth: simulator only


def prepared_paths(processed_dir: Path) -> PreparedPaths:
    return PreparedPaths(
        seed_composites=processed_dir / "seed_composites.parquet",
        candidates=processed_dir / "candidates.parquet",
        candidate_points=processed_dir / "candidate_points.parquet",
        collars=processed_dir / "collars.parquet",
        meta=processed_dir / "meta.json",
        heldout_composites=processed_dir / TRUTH_DIR / "heldout_composites.parquet",
    )


def prepare_tables(tables: DrillholeTables, cfg: Config) -> PreparedPaths:
    rng = np.random.default_rng(cfg.seed if cfg.split.seed is None else cfg.split.seed)
    comps = composite(tables.assays, cfg.composite.length_m, cfg.composite.min_coverage)
    comps = desurvey(tables.collars, tables.surveys, comps)

    # Only holes with at least one composite can be seeds or be scored as candidates.
    collars = tables.collars[tables.collars["hole_id"].isin(comps["hole_id"])]
    seed_ids = choose_seed_holes(collars, cfg.split.n_seed, cfg.split.method, rng)
    heldout_ids = sorted(set(collars["hole_id"]) - set(seed_ids))
    candidates, points = build_candidates(
        collars, tables.surveys, heldout_ids, cfg.composite.length_m
    )
    truth = comps[comps["hole_id"].isin(heldout_ids)].merge(
        candidates[["hole_id", "hole_idx"]], on="hole_id"
    )

    paths = prepared_paths(cfg.data.processed_dir)
    paths.heldout_composites.parent.mkdir(parents=True, exist_ok=True)
    comps[comps["hole_id"].isin(seed_ids)].to_parquet(paths.seed_composites, index=False)
    candidates.to_parquet(paths.candidates, index=False)
    points.to_parquet(paths.candidate_points, index=False)
    collars.to_parquet(paths.collars, index=False)
    truth.to_parquet(paths.heldout_composites, index=False)
    meta = {
        "origin": list(tables.origin),
        "n_holes": len(collars),
        "n_seed": len(seed_ids),
        "n_candidates": len(candidates),
        "n_composites": len(comps),
        "seed_hole_ids": seed_ids,
        "composite_length_m": cfg.composite.length_m,
        "split_method": cfg.split.method,
    }
    paths.meta.write_text(json.dumps(meta, indent=2))
    log.info(
        "prepared holes=%d seed=%d candidates=%d composites=%d dir=%s",
        len(collars),
        len(seed_ids),
        len(candidates),
        len(comps),
        cfg.data.processed_dir,
    )
    return paths


def prepare(cfg: Config) -> PreparedPaths:
    return prepare_tables(load_tables(cfg.data), cfg)
