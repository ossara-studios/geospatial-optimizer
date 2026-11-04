from __future__ import annotations

import pytest
from pydantic import ValidationError

from geoopt.config import Limits, NodeSpec
from geoopt.limits import check_depth
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
