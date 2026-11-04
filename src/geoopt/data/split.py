"""Seed / held-out split and the grade-free candidate table."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from geoopt.data.desurvey import desurvey

SplitMethod = Literal["earliest", "spread"]

CANDIDATE_COLUMNS = ["hole_idx", "hole_id", "x", "y", "z", "azimuth", "dip", "depth"]
CANDIDATE_POINT_COLUMNS = ["hole_idx", "from_m", "to_m", "length_m", "depth", "x", "y", "z"]


def choose_seed_holes(
    collars: pd.DataFrame, n_seed: int, method: SplitMethod, rng: np.random.Generator
) -> list[str]:
    """Pick the holes treated as "already drilled".

    earliest: the n earliest by drill date (requires dates on every hole).
    spread:   farthest-point sampling on collar XY from a random start, so the seed
              set covers the area instead of clustering in one breccia.
    """
    if not 0 < n_seed < len(collars):
        raise ValueError(f"n_seed must be in (0, {len(collars)}), got {n_seed}")
    if method == "earliest":
        if "drill_date" not in collars or collars["drill_date"].isna().any():
            raise ValueError("split.method=earliest needs a drill_date on every hole")
        ordered = collars.sort_values(["drill_date", "hole_id"])
        return ordered["hole_id"].head(n_seed).tolist()

    xy = collars[["x", "y"]].to_numpy(dtype=float)
    chosen = [int(rng.integers(len(xy)))]
    dist = np.linalg.norm(xy - xy[chosen[0]], axis=1)
    for _ in range(n_seed - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))
    return collars["hole_id"].iloc[chosen].tolist()


def build_candidates(
    collars: pd.DataFrame, surveys: pd.DataFrame, hole_ids: list[str], length_m: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate holes and their sample locations, built from geometry only.

    Points are regular bins over each hole's full planned depth, so the candidate set
    reveals nothing about grades or about which intervals were assayed.
    """
    cand = (
        collars[collars["hole_id"].isin(hole_ids)]
        .sort_values("hole_id")
        .reset_index(drop=True)
        .rename(columns={"total_depth": "depth"})
    )
    cand.insert(0, "hole_idx", np.arange(len(cand), dtype=np.uint32))

    bins = []
    for row in cand.itertuples():
        lo = np.arange(0.0, row.depth, length_m)
        hi = np.minimum(lo + length_m, row.depth)
        bins.append(
            pd.DataFrame(
                {"hole_id": row.hole_id, "hole_idx": row.hole_idx, "from_m": lo, "to_m": hi}
            )
        )
    points = pd.concat(bins, ignore_index=True)
    points["length_m"] = points["to_m"] - points["from_m"]
    points["depth"] = (points["from_m"] + points["to_m"]) / 2
    points = desurvey(collars, surveys, points)
    return cand[CANDIDATE_COLUMNS], points[CANDIDATE_POINT_COLUMNS]
