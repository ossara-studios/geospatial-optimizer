"""In-process bridge: same limit logic as EAE, auto-approval, no network. For backtests."""

from __future__ import annotations

import logging

from geoopt.config import Limits
from geoopt.limits import check_recommendation
from geoopt.protocol import CheckResult, CheckStatus, Completion, Decision, RecommendationMsg
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim

log = logging.getLogger(__name__)


class InProcessBridge:
    def __init__(self, limits: Limits, rig: RigSim, lab: LabSim) -> None:
        self._limits = limits
        self._rig = rig
        self._lab = lab
        self._pending: dict[int, tuple[RecommendationMsg, CheckResult]] = {}

    async def submit(self, rec: RecommendationMsg) -> CheckResult:
        check = check_recommendation(
            rec.collar_x, rec.collar_y, rec.azimuth, rec.dip, rec.depth, self._limits
        )
        if check.status is not CheckStatus.REJECTED:
            self._pending[rec.req_id] = (rec, check)
        return check

    async def await_approval(self, req_id: int) -> Decision:
        if req_id not in self._pending:
            raise KeyError(f"no checked request {req_id}")
        return Decision.APPROVED

    async def await_completion(self, req_id: int) -> Completion:
        rec, check = self._pending.pop(req_id)
        metres = self._rig.metres_drilled(rec.hole_idx, check.depth)
        self._lab.release(rec.hole_idx, metres)
        return Completion(req_id, rec.hole_idx, metres)
