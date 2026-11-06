"""Drill-rig simulator: "drills" a historical hole."""

from __future__ import annotations

import pandas as pd


class RigSim:
    def __init__(self, candidates: pd.DataFrame) -> None:
        self._depth = dict(
            zip(candidates["hole_idx"].astype(int), candidates["depth"].astype(float), strict=True)
        )

    def metres_drilled(self, hole_idx: int, dispatched_depth: float) -> float:
        """A historical hole can't go deeper than it was actually drilled."""
        return min(dispatched_depth, self._depth[hole_idx])
