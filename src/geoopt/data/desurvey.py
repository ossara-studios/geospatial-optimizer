"""Minimum-curvature desurvey: downhole depth -> local XYZ.

Conventions (see docs/data_schema.md): azimuth in degrees clockwise from grid north,
dip in degrees with negative = downward (-90 is vertical down). X east, Y north, Z up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def direction(azimuth_deg: np.ndarray, dip_deg: np.ndarray) -> np.ndarray:
    """Unit direction vectors (n, 3) for azimuth/dip pairs."""
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    dip = np.radians(np.asarray(dip_deg, dtype=float))
    return np.stack([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), np.sin(dip)], axis=-1)


def _ratio_factor(dogleg: np.ndarray) -> np.ndarray:
    """2/b * tan(b/2), which tends to 1 as the dogleg b tends to 0."""
    small = dogleg < 1e-9
    safe = np.where(small, 1.0, dogleg)
    return np.where(small, 1.0, 2.0 / safe * np.tan(safe / 2.0))


def _slerp(t1: np.ndarray, t2: np.ndarray, dogleg: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Direction a fraction f along the arc from t1 to t2."""
    small = (dogleg < 1e-9)[:, None]
    safe = np.where(dogleg < 1e-9, 1.0, dogleg)[:, None]
    f = f[:, None]
    arc = (np.sin((1 - f) * safe) * t1 + np.sin(f * safe) * t2) / np.sin(safe)
    return np.where(small, t1, arc)


def desurvey_hole(
    collar: np.ndarray,
    station_depth: np.ndarray,
    azimuth_deg: np.ndarray,
    dip_deg: np.ndarray,
    query_depth: np.ndarray,
) -> np.ndarray:
    """Positions (n, 3) of `query_depth` along one hole by minimum curvature.

    Stations must be sorted by depth. The first orientation applies above the first
    station and the last orientation extends below the last station.
    """
    md = np.asarray(station_depth, dtype=float)
    t = direction(azimuth_deg, dip_deg)
    if md.size == 0:
        raise ValueError("need at least one survey station")
    if md[0] > 0:
        md = np.concatenate([[0.0], md])
        t = np.vstack([t[:1], t])

    # Station positions.
    dmd = np.diff(md)
    dogleg = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0))
    steps = (dmd / 2.0 * _ratio_factor(dogleg))[:, None] * (t[:-1] + t[1:])
    pos = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)]) + np.asarray(collar, dtype=float)

    q = np.asarray(query_depth, dtype=float)
    if np.any(q < 0):
        raise ValueError("query depths must be >= 0")
    seg = np.clip(np.searchsorted(md, q, side="right") - 1, 0, len(md) - 1)
    out = np.empty((q.size, 3))

    # Beyond the last station: straight line along the last direction.
    beyond = seg == len(md) - 1
    out[beyond] = pos[-1] + (q[beyond] - md[-1])[:, None] * t[-1]

    inner = ~beyond
    if inner.any():
        s = seg[inner]
        d = q[inner] - md[s]
        f = d / dmd[s]
        b = dogleg[s] * f
        t_q = _slerp(t[s], t[s + 1], dogleg[s], f)
        out[inner] = pos[s] + (d / 2.0 * _ratio_factor(b))[:, None] * (t[s] + t_q)
    return out


def desurvey(collars: pd.DataFrame, surveys: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Add x, y, z to `points` (hole_id, depth) using collars and surveys.

    Holes without survey stations fall back to a straight line using the collar's
    azimuth/dip.
    """
    collars = collars.set_index("hole_id")
    by_hole = dict(tuple(surveys.sort_values(["hole_id", "depth"]).groupby("hole_id")))
    parts = []
    for hole_id, pts in points.groupby("hole_id", sort=False):
        c = collars.loc[hole_id]
        xyz0 = np.array([c["x"], c["y"], c["z"]], dtype=float)
        sv = by_hole.get(hole_id)
        if sv is None or sv.empty:
            md, az, dip = np.array([0.0]), np.array([c["azimuth"]]), np.array([c["dip"]])
        else:
            md, az, dip = sv["depth"].to_numpy(), sv["azimuth"].to_numpy(), sv["dip"].to_numpy()
        xyz = desurvey_hole(xyz0, md, az, dip, pts["depth"].to_numpy())
        parts.append(pts.assign(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
    if not parts:
        return points.assign(x=np.nan, y=np.nan, z=np.nan)
    return pd.concat(parts).loc[points.index]
