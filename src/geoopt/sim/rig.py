"""Drill-rig simulator: "drills" a historical hole and emits synthetic telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Telemetry:
    """SYNTHETIC rig telemetry for the HMI, not derived from any real measurement."""

    depth_m: float
    torque_knm: float
    water_lpm: float
    power_kw: float


class RigSim:
    def __init__(self, candidates: pd.DataFrame, seed: int) -> None:
        self._depth = dict(
            zip(candidates["hole_idx"].astype(int), candidates["depth"].astype(float), strict=True)
        )
        self._rng = np.random.default_rng(seed)

    def metres_drilled(self, hole_idx: int, dispatched_depth: float) -> float:
        """A historical hole can't go deeper than it was actually drilled."""
        return min(dispatched_depth, self._depth[hole_idx])

    def telemetry(self, depth_m: float) -> Telemetry:
        noise = self._rng.normal(0, 1, 3)
        return Telemetry(
            depth_m=depth_m,
            torque_knm=round(2.0 + 0.004 * depth_m + 0.2 * noise[0], 2),
            water_lpm=round(40.0 + 3.0 * noise[1], 1),
            power_kw=round(150.0 + 0.1 * depth_m + 8.0 * noise[2], 1),
        )
