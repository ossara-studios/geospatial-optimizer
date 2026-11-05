"""Block model: a regular grid of block centres near drilling, and grade-above-cutoff calls.

Blocks are classified the way a resource estimate is: ore when the estimated (expected) grade
is above the cutoff. The estimate is the lognormal mean of the latent log grade at the block
centre (no nugget: a block averages it out).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from geoopt.metrics import classification


def block_centres(near_xyz: np.ndarray, size_m: tuple[float, float, float], radius_m: float):
    """Centres of a regular grid over `near_xyz`'s extent, kept within `radius_m` of a point."""
    near_xyz = np.asarray(near_xyz, dtype=float)
    size = np.asarray(size_m, dtype=float)
    lo = np.floor(near_xyz.min(axis=0) / size) * size
    hi = near_xyz.max(axis=0)
    axes = [np.arange(lo[i] + size[i] / 2, hi[i] + size[i], size[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d, _ = cKDTree(near_xyz).query(grid, distance_upper_bound=radius_m)
    return grid[np.isfinite(d)]


def expected_grade(mean_log: np.ndarray, std_log: np.ndarray, log_eps: float) -> np.ndarray:
    """E[Cu] (%) for log(Cu + eps) ~ N(mean, std^2)."""
    return np.exp(mean_log + std_log**2 / 2) - log_eps


def boundary_log_level(cutoff_pct: float, std_log: np.ndarray, log_eps: float) -> np.ndarray:
    """Latent mean at which E[Cu] equals the cutoff, given the latent std."""
    return np.log(cutoff_pct + log_eps) - std_log**2 / 2


def block_agreement(
    grade: np.ndarray, ref_grade: np.ndarray, cutoff_pct: float
) -> dict[str, float]:
    """A block estimate scored against the reference estimate.

    f1/bacc: ore-vs-waste block calls. ore_volume_err / metal_err: signed relative error in
    the number of ore blocks and in contained Cu (sum of grade over ore blocks).
    """
    ore, ref_ore = grade > cutoff_pct, ref_grade > cutoff_pct
    c = classification(ore, ref_ore)
    ref_n, ref_metal = ref_ore.sum(), ref_grade[ref_ore].sum()
    return {
        "block_f1": c["f1"],
        "block_bacc": c["balanced_accuracy"],
        "block_ore_volume_err": float((ore.sum() - ref_n) / ref_n) if ref_n else float("nan"),
        "block_metal_err": float((grade[ore].sum() - ref_metal) / ref_metal)
        if ref_metal
        else float("nan"),
    }
