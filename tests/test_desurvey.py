from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoopt.data.desurvey import desurvey, desurvey_hole, direction

COLLAR = np.array([100.0, 200.0, 50.0])


def test_vertical_hole() -> None:
    xyz = desurvey_hole(COLLAR, [0.0], [0.0], [-90.0], [0.0, 10.0, 250.0])
    np.testing.assert_allclose(xyz, [[100, 200, 50], [100, 200, 40], [100, 200, -200]], atol=1e-9)


def test_straight_inclined_hole() -> None:
    # Dip -60 toward east: per metre, east cos60, down sin60.
    xyz = desurvey_hole(COLLAR, [0.0, 50.0], [90.0, 90.0], [-60.0, -60.0], [100.0])
    expected = COLLAR + 100.0 * np.array([0.5, 0.0, -np.sqrt(3) / 2])
    np.testing.assert_allclose(xyz[0], expected, atol=1e-9)


def test_constant_curvature_arc_matches_circle() -> None:
    # Builds from vertical (-90) to horizontal (0) over a quarter circle of radius R:
    # arc length = pi R / 2; end point R east and R down.
    radius = 100.0
    length = np.pi * radius / 2
    xyz = desurvey_hole([0, 0, 0], [0.0, length], [90.0, 90.0], [-90.0, 0.0], [length / 2, length])
    np.testing.assert_allclose(xyz[1], [radius, 0, -radius], atol=1e-6)
    # Midpoint lies on the same circle (centre at (R, 0, 0)).
    assert np.linalg.norm(xyz[0] - np.array([radius, 0, 0])) == pytest.approx(radius, abs=1e-6)


def test_direction_is_unit() -> None:
    v = direction(np.array([0, 45, 200]), np.array([-90, -45, -10]))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)


def test_frame_api_with_missing_surveys_uses_collar_orientation() -> None:
    collars = pd.DataFrame(
        {
            "hole_id": ["A", "B"],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "azimuth": [0.0, 0.0],
            "dip": [-90.0, -90.0],
        }
    )
    surveys = pd.DataFrame({"hole_id": ["A"], "depth": [0.0], "azimuth": [0.0], "dip": [-90.0]})
    points = pd.DataFrame({"hole_id": ["B", "A"], "depth": [5.0, 3.0]})
    out = desurvey(collars, surveys, points)
    np.testing.assert_allclose(
        out[["x", "y", "z"]].to_numpy(), [[10, 0, -5], [0, 0, -3]], atol=1e-9
    )
