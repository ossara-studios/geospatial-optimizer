"""Pure limit checks, mirrored by the EAE limit-check function block."""

from __future__ import annotations

from geoopt.config import Limits
from geoopt.protocol import CheckResult, CheckStatus, Reason


def check_depth(depth: float, limits: Limits) -> CheckResult:
    """Clip planned depth to the site maximum."""
    if depth > limits.max_depth_m:
        return CheckResult(CheckStatus.CLIPPED, Reason.DEPTH_CLIPPED, limits.max_depth_m)
    return CheckResult(CheckStatus.ACCEPTED, Reason.NONE, depth)
