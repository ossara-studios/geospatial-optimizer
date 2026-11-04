"""Fixed-length downhole compositing (length-weighted grade)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPOSITE_COLUMNS = ["hole_id", "from_m", "to_m", "length_m", "sampled_m", "cu", "depth"]


def composite_hole(
    from_m: np.ndarray,
    to_m: np.ndarray,
    cu: np.ndarray,
    length_m: float,
    min_coverage: float,
) -> pd.DataFrame:
    """Composite one hole's intervals into bins [k*L, (k+1)*L) from the collar.

    Intervals with NaN grade (not analysed) carry no weight. A composite is kept when
    its assayed length is at least `min_coverage` of its own length; the last bin is
    truncated at the deepest interval end.
    """
    start, end, grade = (np.asarray(a, dtype=float) for a in (from_m, to_m, cu))
    assayed = ~np.isnan(grade)
    start, end, grade = start[assayed], end[assayed], grade[assayed]
    if start.size == 0:
        return pd.DataFrame(columns=COMPOSITE_COLUMNS[1:])

    hole_end = end.max()
    lo = np.arange(0.0, hole_end, length_m)
    hi = np.minimum(lo + length_m, hole_end)
    overlap = np.clip(
        np.minimum(hi[:, None], end[None, :]) - np.maximum(lo[:, None], start[None, :]), 0.0, None
    )
    sampled = overlap.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_grade = (overlap * grade[None, :]).sum(axis=1) / sampled

    keep = (sampled > 0) & (sampled >= min_coverage * (hi - lo))
    return pd.DataFrame(
        {
            "from_m": lo[keep],
            "to_m": hi[keep],
            "length_m": (hi - lo)[keep],
            "sampled_m": sampled[keep],
            "cu": bin_grade[keep],
            "depth": ((lo + hi) / 2)[keep],
        }
    )


def composite(assays: pd.DataFrame, length_m: float, min_coverage: float) -> pd.DataFrame:
    """Composite all holes. `assays` has hole_id, from_m, to_m, cu."""
    parts = [
        composite_hole(g["from_m"], g["to_m"], g["cu"], length_m, min_coverage).assign(
            hole_id=hole_id
        )
        for hole_id, g in assays.groupby("hole_id", sort=True)
    ]
    if not parts:
        return pd.DataFrame(columns=COMPOSITE_COLUMNS)
    return pd.concat(parts, ignore_index=True)[COMPOSITE_COLUMNS]
