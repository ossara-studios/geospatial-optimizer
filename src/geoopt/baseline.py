"""Near-ore baseline: a no-model rule a geologist might use, replayed on the prepared data.

Each step drills the eligible hole with the largest share of its trace (clipped to the rig's
maximum depth) within `near_ore_m` of an ore composite seen so far. It is scored with the same
live discovery KPI as the planner, so `geoopt run-loop` can report metres and cost saved against
it. Evaluation side: it reads held-out assays directly; the planner never imports it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from geoopt.config import Config
from geoopt.data.prepare import prepared_paths
from geoopt.kpi import DiscoveryTracker, kpi_geometry, write_baseline_meta
from geoopt.limits import check_recommendation
from geoopt.protocol import CheckStatus

log = logging.getLogger(__name__)


def near_ore_curve(cfg: Config, out_path: Path) -> pd.DataFrame:
    paths = prepared_paths(cfg.data.processed_dir)
    cand = pd.read_parquet(paths.candidates).set_index("hole_idx", drop=False)
    pts = pd.read_parquet(paths.candidate_points)
    seed = pd.read_parquet(paths.seed_composites)
    lim = cfg.limits
    ok = [
        check_recommendation(r.x, r.y, r.azimuth, r.dip, r.depth, lim).status
        is not CheckStatus.REJECTED
        for r in cand.itertuples()
    ]
    cand = cand[ok]
    depth = cand["depth"].clip(upper=lim.max_depth_m)
    pts = pts[pts["hole_idx"].isin(cand.index) & (pts["to_m"] <= lim.max_depth_m + 1e-6)]
    truth = pd.read_parquet(paths.heldout_composites)
    by_hole = {h: g for h, g in truth.groupby("hole_idx")}
    geometry = kpi_geometry(pd.read_parquet(paths.candidate_points), seed, lim.max_depth_m)
    tracker = DiscoveryTracker(
        cfg.kpi, cfg.model.cutoff_pct, cfg.backtest.block_size_m, geometry, seed
    )
    known = [seed.loc[seed["cu"] > cfg.model.cutoff_pct, ["x", "y", "z"]].to_numpy()]
    pxyz, phole, plen = pts[["x", "y", "z"]].to_numpy(), pts["hole_idx"].to_numpy(), pts["length_m"]
    drilled: set[int] = set()
    rows = [{"metres": 0.0, "new_ore_blocks": 0, "holes": 0}]
    budget = cfg.kpi.baseline_budget_m
    while tracker.kpis().metres < budget:
        d, _ = cKDTree(np.vstack(known)).query(pxyz, distance_upper_bound=cfg.kpi.near_ore_m)
        share = pd.Series(np.isfinite(d) * plen.to_numpy(), index=phole).groupby(level=0).sum()
        share = (share / depth.reindex(share.index)).drop(list(drilled & set(share.index)))
        if share.empty:
            break
        h = int(share.idxmax())
        drilled.add(h)
        g = by_hole.get(h, truth.iloc[:0])
        g = g[g["to_m"] <= depth[h] + 1e-9]
        k = tracker.add_hole(float(depth[h]), g)
        known.append(g.loc[g["cu"] > cfg.model.cutoff_pct, ["x", "y", "z"]].to_numpy())
        rows.append({"metres": k.metres, "new_ore_blocks": k.new_ore_blocks, "holes": k.holes})
    curve = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(out_path, index=False)
    write_baseline_meta(
        out_path,
        {
            "rule": f"near-ore: largest trace share within {cfg.kpi.near_ore_m:g} m of known ore",
            "processed_dir": str(cfg.data.processed_dir),
            "budget_m": budget,
            "holes": int(curve["holes"].iloc[-1]),
            "new_ore_blocks": int(curve["new_ore_blocks"].iloc[-1]),
        },
    )
    log.info(
        "near-ore baseline %d holes %.0f m -> %s", len(drilled), curve.metres.iloc[-1], out_path
    )
    return curve
