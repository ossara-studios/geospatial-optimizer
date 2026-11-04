from __future__ import annotations

import pytest
from pydantic import ValidationError

from geoopt.config import Limits, NodeSpec
from geoopt.limits import check_depth, check_recommendation
from geoopt.protocol import CheckStatus, Reason

LIMITS = Limits(max_depth_m=500.0)


@pytest.mark.parametrize("depth", [1.0, 300.0, 500.0])
def test_depth_within_limit_is_accepted(depth: float) -> None:
    result = check_depth(depth, LIMITS)
    assert result.status is CheckStatus.ACCEPTED
    assert result.reason is Reason.NONE
    assert result.depth == depth


def test_depth_over_limit_is_clipped() -> None:
    result = check_depth(650.0, LIMITS)
    assert result.status is CheckStatus.CLIPPED
    assert result.reason is Reason.DEPTH_CLIPPED
    assert result.depth == 500.0


def test_node_spec_needs_exactly_one_identifier() -> None:
    with pytest.raises(ValidationError):
        NodeSpec(type="UInt32")
    with pytest.raises(ValidationError):
        NodeSpec(id="a", node_id="ns=2;s=a", type="UInt32")


FULL = Limits(
    max_depth_m=500.0,
    site_bounds={"x_min": 0, "x_max": 1000, "y_min": 0, "y_max": 1000},
    no_go_zones=[
        {"kind": "rect", "name": "drainage", "x_min": 100, "x_max": 200, "y_min": 0, "y_max": 1000},
        {"kind": "circle", "name": "heritage", "x": 800, "y": 800, "radius": 50},
    ],
    dip_range=(-90.0, -45.0),
    azimuth_range=(0.0, 360.0),
)


@pytest.mark.parametrize(
    ("x", "y", "azimuth", "dip", "depth", "status", "reason", "out_depth"),
    [
        (500, 500, 90, -60, 300, CheckStatus.ACCEPTED, Reason.NONE, 300),
        (500, 500, 90, -60, 650, CheckStatus.CLIPPED, Reason.DEPTH_CLIPPED, 500),
        (1500, 500, 90, -60, 300, CheckStatus.REJECTED, Reason.OUT_OF_BOUNDS, 300),
        (150, 500, 90, -60, 300, CheckStatus.REJECTED, Reason.NO_GO_ZONE, 300),
        (820, 810, 90, -60, 300, CheckStatus.REJECTED, Reason.NO_GO_ZONE, 300),
        (500, 500, 90, -30, 300, CheckStatus.REJECTED, Reason.DIP_OUT_OF_RANGE, 300),
        (500, 500, 400, -60, 300, CheckStatus.REJECTED, Reason.AZIMUTH_OUT_OF_RANGE, 300),
        # Rejections win over clipping.
        (150, 500, 90, -60, 650, CheckStatus.REJECTED, Reason.NO_GO_ZONE, 650),
    ],
)
def test_every_reason_code(x, y, azimuth, dip, depth, status, reason, out_depth) -> None:
    result = check_recommendation(x, y, azimuth, dip, depth, FULL)
    assert (result.status, result.reason, result.depth) == (status, reason, out_depth)
