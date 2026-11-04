"""Coordinate and grade transforms, kept in one place so they invert consistently."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoordScaler:
    """Standardize local-metre coordinates for the GP."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, xyz: np.ndarray) -> CoordScaler:
        xyz = np.asarray(xyz, dtype=float)
        std = xyz.std(axis=0)
        return cls(xyz.mean(axis=0), np.where(std > 0, std, 1.0))

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=float) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.std + self.mean

    def lengthscales_m(self, scaled: np.ndarray) -> np.ndarray:
        """Convert kernel lengthscales from scaled units back to metres per axis."""
        return np.asarray(scaled, dtype=float) * self.std


def to_log_grade(cu: np.ndarray, eps: float) -> np.ndarray:
    cu = np.asarray(cu, dtype=float)
    if np.any(cu < 0) or np.any(np.isnan(cu)):
        raise ValueError("grades must be finite and >= 0 before the log transform")
    return np.log(cu + eps)


def from_log_grade(y: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(np.asarray(y, dtype=float)) - eps
