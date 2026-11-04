from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoopt.data.split import build_candidates, choose_seed_holes


def _collars(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "hole_id": [f"H{i:02d}" for i in range(n)],
            "x": rng.uniform(0, 1000, n),
            "y": rng.uniform(0, 1000, n),
            "z": np.zeros(n),
            "azimuth": np.zeros(n),
            "dip": np.full(n, -90.0),
            "total_depth": rng.uniform(80, 300, n),
            "drill_date": pd.date_range("1960-01-01", periods=n, freq="YS")[::-1],
        }
    )


def test_spread_split_is_deterministic_and_unique() -> None:
    collars = _collars()
    a = choose_seed_holes(collars, 4, "spread", np.random.default_rng(7))
    b = choose_seed_holes(collars, 4, "spread", np.random.default_rng(7))
    assert a == b
    assert len(set(a)) == 4


def test_earliest_split_uses_dates() -> None:
    collars = _collars()
    assert choose_seed_holes(collars, 3, "earliest", np.random.default_rng(0)) == [
        "H11",
        "H10",
        "H09",
    ]


def test_earliest_split_requires_dates() -> None:
    collars = _collars().drop(columns="drill_date")
    with pytest.raises(ValueError, match="drill_date"):
        choose_seed_holes(collars, 3, "earliest", np.random.default_rng(0))


def test_candidate_tables_have_no_grade_information() -> None:
    collars = _collars()
    surveys = pd.DataFrame(columns=["hole_id", "depth", "azimuth", "dip"])
    cand, points = build_candidates(collars, surveys, ["H03", "H01"], length_m=10.0)
    forbidden = {"cu", "grade", "sampled_m", "mo", "au", "ag"}
    assert not forbidden & {c.lower() for c in cand.columns}
    assert not forbidden & {c.lower() for c in points.columns}
    assert cand["hole_id"].tolist() == ["H01", "H03"]
    assert cand["hole_idx"].tolist() == [0, 1]
    # Points cover each hole's full planned depth.
    for row in cand.itertuples():
        pts = points[points["hole_idx"] == row.hole_idx]
        assert pts["length_m"].sum() == pytest.approx(row.depth)
