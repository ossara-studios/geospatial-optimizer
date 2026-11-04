from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoopt.data.composite import composite, composite_hole


def _metal(df: pd.DataFrame, length_col: str) -> float:
    return float((df["cu"] * df[length_col]).sum())


def test_conserves_length_weighted_grade() -> None:
    rng = np.random.default_rng(0)
    edges = np.cumsum(rng.uniform(0.5, 4.0, size=60))
    raw = pd.DataFrame(
        {"hole_id": "H1", "from_m": edges[:-1], "to_m": edges[1:], "cu": rng.lognormal(-1, 1, 59)}
    )
    raw["length_m"] = raw["to_m"] - raw["from_m"]
    comps = composite(raw, length_m=10.0, min_coverage=0.0)
    assert _metal(comps, "sampled_m") == pytest.approx(_metal(raw, "length_m"))
    assert comps["sampled_m"].sum() == pytest.approx(raw["length_m"].sum())
    mean_raw = _metal(raw, "length_m") / raw["length_m"].sum()
    assert _metal(comps, "sampled_m") / comps["sampled_m"].sum() == pytest.approx(mean_raw)


def test_split_interval_and_truncated_last_bin() -> None:
    out = composite_hole([0.0, 7.0], [7.0, 13.0], [1.0, 3.0], length_m=10.0, min_coverage=0.0)
    # Bin 0-10: 7 m at 1.0 + 3 m at 3.0 -> 1.6. Bin 10-13: 3 m at 3.0.
    np.testing.assert_allclose(out["cu"], [1.6, 3.0])
    np.testing.assert_allclose(out["to_m"], [10.0, 13.0])
    np.testing.assert_allclose(out["depth"], [5.0, 11.5])


def test_unassayed_intervals_are_not_zero_filled() -> None:
    out = composite_hole([0.0, 5.0], [5.0, 10.0], [2.0, np.nan], length_m=10.0, min_coverage=0.0)
    assert out["cu"].iloc[0] == pytest.approx(2.0)
    assert out["sampled_m"].iloc[0] == pytest.approx(5.0)


def test_min_coverage_drops_sparse_bins() -> None:
    out = composite_hole([0.0, 15.0], [10.0, 17.0], [1.0, 1.0], length_m=10.0, min_coverage=0.5)
    # Bin 10-17 has only 2 of 7 m assayed (< 50%), so only bin 0-10 survives.
    np.testing.assert_allclose(out["from_m"], [0.0])
