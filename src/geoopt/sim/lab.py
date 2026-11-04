"""Lab simulator: the only owner of held-out (ground-truth) assays.

The planner never imports this module. It sees assays only as files that LabSim drops
into `runs/<run_id>/lab/<HoleIdx>.parquet` after a hole is reported complete.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LAB_COLUMNS = ["hole_idx", "hole_id", "from_m", "to_m", "length_m", "x", "y", "z", "cu"]


def lab_path(lab_dir: Path, hole_idx: int) -> Path:
    return lab_dir / f"{hole_idx}.parquet"


class LabSim:
    def __init__(self, truth_path: Path, lab_dir: Path) -> None:
        self.__truth = pd.read_parquet(truth_path)  # name-mangled: private by convention
        self._lab_dir = lab_dir
        lab_dir.mkdir(parents=True, exist_ok=True)

    def release(self, hole_idx: int, metres_drilled: float) -> Path:
        """Write the composites of `hole_idx` that lie fully above `metres_drilled`."""
        t = self.__truth
        rows = t[(t["hole_idx"] == hole_idx) & (t["to_m"] <= metres_drilled + 1e-9)]
        path = lab_path(self._lab_dir, hole_idx)
        rows[LAB_COLUMNS].to_parquet(path, index=False)
        log.info("lab released hole_idx=%s composites=%d to %s", hole_idx, len(rows), path)
        return path

    def truth_for_scoring(self) -> pd.DataFrame:
        """Held-out composites for backtest metrics only (never for the planner)."""
        return self.__truth[LAB_COLUMNS].copy()
