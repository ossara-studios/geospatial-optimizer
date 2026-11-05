"""Live discovery KPIs: only new ore counts, and cost saved reads the baseline curve."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from geoopt.config import KpiConfig
from geoopt.kpi import DiscoveryTracker, baseline_metres_for


def _comps(xyz: list[tuple[float, float, float]], cu: list[float]) -> pd.DataFrame:
    a = np.asarray(xyz, dtype=float)
    return pd.DataFrame({"x": a[:, 0], "y": a[:, 1], "z": a[:, 2], "cu": cu, "length_m": 10.0})


def _tracker(baseline: pd.DataFrame | None = None) -> DiscoveryTracker:
    geometry = np.array([[x, 0.0, 0.0] for x in np.arange(0, 1000, 5.0)])
    seed = _comps([(0, 0, 0)], [1.0])  # ore at the origin is already known
    cfg = KpiConfig(radius_m=20.0, block_radius_m=30.0, density_t_m3=2.0, drill_cost_aud_per_m=100)
    return DiscoveryTracker(cfg, 0.2, (25.0, 25.0, 10.0), geometry, seed, baseline)


def test_known_ore_and_waste_add_nothing() -> None:
    t = _tracker()
    k = t.add_hole(100.0, _comps([(0, 0, 0), (500, 0, 0)], [2.0, 0.05]))
    assert k.new_ore_blocks == 0  # seed already reached the origin; 500 m sample is waste
    assert k.ore_m == 10.0
    assert k.drill_cost_aud_m == 100.0 * 100 / 1e6


def test_new_ore_counted_once() -> None:
    t = _tracker()
    first = t.add_hole(50.0, _comps([(500, 0, 0)], [1.0]))
    assert first.new_ore_blocks > 0
    again = t.add_hole(50.0, _comps([(501, 0, 0)], [1.0]))
    assert again.new_ore_blocks == first.new_ore_blocks  # re-drilling known ore
    assert math.isclose(again.new_ore_mt, first.new_ore_blocks * 25 * 25 * 10 * 2.0 / 1e6)
    assert again.new_cu_kt > 0


def test_cost_saved_against_baseline() -> None:
    curve = pd.DataFrame({"metres": [0.0, 200.0, 400.0], "new_ore_blocks": [0, 1, 50]})
    assert baseline_metres_for(curve, 1) == 200.0
    assert math.isnan(baseline_metres_for(curve, 51))
    t = _tracker(curve)
    k = t.add_hole(120.0, _comps([(500, 0, 0)], [1.0]))
    assert k.baseline_metres == 200.0 if k.new_ore_blocks == 1 else 400.0
    assert math.isclose(k.metres_saved, k.baseline_metres - 120.0)
    assert math.isclose(k.cost_saved_aud_m, k.metres_saved * 100 / 1e6)


def test_assays_can_arrive_after_the_hole() -> None:
    t = _tracker()
    k = t.add_hole(80.0, None)
    assert (k.holes, k.metres, k.new_ore_blocks) == (1, 80.0, 0)
    t.add_assays(_comps([(500, 0, 0)], [1.0]))
    assert t.kpis().new_ore_blocks > 0
