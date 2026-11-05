from __future__ import annotations

import numpy as np
import pytest

from geoopt.blocks import block_agreement, block_centres, boundary_log_level, expected_grade


def test_block_centres_stay_near_data() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    blocks = block_centres(pts, (25.0, 25.0, 10.0), 30.0)
    d = np.min(np.linalg.norm(blocks[:, None, :] - pts[None], axis=-1), axis=1)
    assert (d <= 30.0).all() and len(blocks) > 0
    assert not ((blocks[:, 0] > 60) & (blocks[:, 0] < 140)).any()  # gap between the points


def test_expected_grade_and_boundary() -> None:
    eps, cut, std = 0.001, 0.2, np.array([0.5, 1.0])
    level = boundary_log_level(cut, std, eps)
    np.testing.assert_allclose(expected_grade(level, std, eps), cut)


def test_block_agreement() -> None:
    ref = np.array([0.5, 0.1, 0.3, 0.05])
    perfect = block_agreement(ref, ref, 0.2)
    assert perfect["block_f1"] == 1.0 and perfect["block_metal_err"] == 0.0
    over = block_agreement(np.array([0.5, 0.3, 0.3, 0.05]), ref, 0.2)
    assert over["block_ore_volume_err"] == pytest.approx(0.5)
    assert over["block_metal_err"] == pytest.approx(0.3 / 0.8)
