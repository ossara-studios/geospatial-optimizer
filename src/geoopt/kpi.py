"""Live discovery KPIs, computed only from assays the lab has released (planner-visible).

New ore: blocks of a regular grid (25 x 25 x 10 m by default) within `discovery_radius_m` of an
assayed ore composite (Cu > cutoff) that no earlier ore composite (seed holes included) was
within reach of. Re-drilling known ore adds nothing. Tonnes use an assumed bulk density;
contained Cu uses the mean grade of the ore composites near each new block. Cost saved compares
the metres drilled with the metres the near-ore baseline needed to find as much new ore
(`geoopt baseline`), at an assumed all-in drilling cost per metre. All of it is INDICATIVE.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from geoopt.blocks import block_centres
from geoopt.config import KpiConfig


@dataclass(frozen=True)
class Kpis:
    holes: int = 0
    metres: float = 0.0
    ore_m: float = 0.0  # assayed metres above cutoff
    new_ore_blocks: int = 0
    new_ore_mt: float = 0.0
    new_cu_kt: float = 0.0
    drill_cost_aud_m: float = 0.0
    baseline_metres: float = float("nan")  # near-ore rule's metres to find as much new ore
    metres_saved: float = float("nan")
    cost_saved_aud_m: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def kpi_geometry(points: pd.DataFrame, seed: pd.DataFrame, max_depth_m: float) -> np.ndarray:
    """Where blocks can be: candidate traces down to the rig's maximum depth, plus seed samples."""
    pts = points[points["to_m"] <= max_depth_m + 1e-6]
    return np.vstack([pts[["x", "y", "z"]].to_numpy(), seed[["x", "y", "z"]].to_numpy()])


def load_baseline(path: Path | None) -> pd.DataFrame | None:
    """Baseline discovery curve (metres, new_ore_blocks), or None when it hasn't been built."""
    if path is None or not Path(path).exists():
        return None
    return pd.read_parquet(path).sort_values("metres")


def baseline_metres_for(curve: pd.DataFrame | None, blocks: int) -> float:
    """Metres the baseline needed to reach `blocks` new ore blocks (NaN if it never did)."""
    if curve is None or blocks <= 0:
        return float("nan")
    hit = curve[curve["new_ore_blocks"] >= blocks]
    return float(hit["metres"].iloc[0]) if len(hit) else float("nan")


class DiscoveryTracker:
    def __init__(
        self,
        cfg: KpiConfig,
        cutoff_pct: float,
        block_size_m: tuple[float, float, float],
        geometry_xyz: np.ndarray,
        seed_composites: pd.DataFrame,
        baseline: pd.DataFrame | None = None,
    ) -> None:
        self._cfg = cfg
        self._cutoff = cutoff_pct
        self._block_t = float(np.prod(block_size_m)) * cfg.density_t_m3
        self._centres = block_centres(geometry_xyz, block_size_m, cfg.block_radius_m)
        self._tree = cKDTree(self._centres)
        self._baseline = baseline
        self._known: set[int] = set()
        self._new: set[int] = set()
        self._grade_sum = np.zeros(len(self._centres))
        self._grade_n = np.zeros(len(self._centres))
        self._holes = 0
        self._metres = 0.0
        self._ore_m = 0.0
        self._known = self._reach(self._ore(seed_composites))

    def _ore(self, comps: pd.DataFrame) -> pd.DataFrame:
        return comps[comps["cu"] > self._cutoff] if len(comps) else comps

    def _reach(self, ore: pd.DataFrame) -> set[int]:
        if not len(ore):
            return set()
        out: set[int] = set()
        for lst in self._tree.query_ball_point(ore[["x", "y", "z"]].to_numpy(), self._cfg.radius_m):
            out.update(lst)
        return out

    def add_hole(self, metres: float, assays: pd.DataFrame | None) -> Kpis:
        """A hole was drilled; `assays` are its released composites (None while at the lab)."""
        self._holes += 1
        self._metres += metres
        if assays is not None:
            self.add_assays(assays)
        return self.kpis()

    def add_assays(self, assays: pd.DataFrame) -> None:
        ore = self._ore(assays)
        if not len(ore):
            return
        self._ore_m += float(ore["length_m"].sum())
        xyz, cu = ore[["x", "y", "z"]].to_numpy(), ore["cu"].to_numpy()
        for lst, g in zip(self._tree.query_ball_point(xyz, self._cfg.radius_m), cu, strict=True):
            for b in lst:
                self._grade_sum[b] += g
                self._grade_n[b] += 1
                if b not in self._known:
                    self._new.add(b)
        self._known |= self._new

    def kpis(self) -> Kpis:
        new = np.fromiter(self._new, dtype=int, count=len(self._new))
        tonnes = len(new) * self._block_t
        grade = self._grade_sum[new] / np.maximum(self._grade_n[new], 1) if len(new) else []
        cu_t = float(np.sum(grade) / 100 * self._block_t) if len(new) else 0.0
        base = baseline_metres_for(self._baseline, len(new))
        saved = base - self._metres if np.isfinite(base) else float("nan")
        return Kpis(
            holes=self._holes,
            metres=self._metres,
            ore_m=self._ore_m,
            new_ore_blocks=len(new),
            new_ore_mt=tonnes / 1e6,
            new_cu_kt=cu_t / 1e3,
            drill_cost_aud_m=self._metres * self._cfg.drill_cost_aud_per_m / 1e6,
            baseline_metres=base,
            metres_saved=saved,
            cost_saved_aud_m=saved * self._cfg.drill_cost_aud_per_m / 1e6
            if np.isfinite(saved)
            else float("nan"),
        )


def write_baseline_meta(path: Path, meta: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
