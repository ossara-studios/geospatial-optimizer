"""Pure limit checks on a recommended hole.

Order: site bounds, no-go zones, dip, azimuth (all REJECT), then depth (CLIP).
A collar can't be moved in replay mode (a moved hole doesn't exist in the dataset),
so location problems reject; depth over the maximum is clipped.
"""

from __future__ import annotations

import math

from geoopt.config import CircleZone, Limits, RectZone
from geoopt.protocol import CheckResult, CheckStatus, Reason


def check_depth(depth: float, limits: Limits) -> CheckResult:
    """Clip planned depth to the site maximum."""
    if depth > limits.max_depth_m:
        return CheckResult(CheckStatus.CLIPPED, Reason.DEPTH_CLIPPED, limits.max_depth_m)
    return CheckResult(CheckStatus.ACCEPTED, Reason.NONE, depth)


def _in_zone(x: float, y: float, zone: RectZone | CircleZone) -> bool:
    if isinstance(zone, RectZone):
        return zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max
    return math.hypot(x - zone.x, y - zone.y) <= zone.radius


def check_recommendation(
    x: float, y: float, azimuth: float, dip: float, depth: float, limits: Limits
) -> CheckResult:
    def reject(reason: Reason) -> CheckResult:
        return CheckResult(CheckStatus.REJECTED, reason, depth)

    b = limits.site_bounds
    if b is not None and not (b.x_min <= x <= b.x_max and b.y_min <= y <= b.y_max):
        return reject(Reason.OUT_OF_BOUNDS)
    if any(_in_zone(x, y, zone) for zone in limits.no_go_zones):
        return reject(Reason.NO_GO_ZONE)
    if not limits.dip_range[0] <= dip <= limits.dip_range[1]:
        return reject(Reason.DIP_OUT_OF_RANGE)
    if not limits.azimuth_range[0] <= azimuth <= limits.azimuth_range[1]:
        return reject(Reason.AZIMUTH_OUT_OF_RANGE)
    return check_depth(depth, limits)
