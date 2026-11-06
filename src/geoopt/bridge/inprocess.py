"""In-process bridge: limit check, auto-approval, simulated rig and lab.

Instant by default (backtests). With `progress` set it runs at demo pace instead: the rig
"drills" at `metres_per_second` and the cycle pauses briefly at each stage, so the live view
can follow it.
"""

from __future__ import annotations

import asyncio
import logging

from geoopt.config import Limits
from geoopt.limits import check_recommendation
from geoopt.progress import Progress
from geoopt.protocol import CheckResult, CheckStatus, Completion, Decision, RecommendationMsg
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim

log = logging.getLogger(__name__)

DRILL_TICK_S = 0.1


class InProcessBridge:
    def __init__(
        self,
        limits: Limits,
        rig: RigSim,
        lab: LabSim,
        progress: Progress | None = None,
        metres_per_second: float = 60.0,
    ) -> None:
        self._limits = limits
        self._rig = rig
        self._lab = lab
        self._progress = progress
        self._mps = metres_per_second
        self._pending: dict[int, tuple[RecommendationMsg, CheckResult]] = {}

    async def _pause(self) -> None:
        if self._progress is not None:
            await asyncio.sleep(self._progress.pause_s)

    async def next_req_id(self) -> int:
        return 1

    async def submit(self, rec: RecommendationMsg) -> CheckResult:
        check = check_recommendation(
            rec.collar_x, rec.collar_y, rec.azimuth, rec.dip, rec.depth, self._limits
        )
        if self._progress is not None:
            rejected = check.status is CheckStatus.REJECTED
            self._progress.at("check", req_id=rec.req_id, hole_idx=rec.hole_idx,
                              rejected=rejected, depth_m=0.0, target_m=0.0)  # fmt: skip
        await self._pause()
        if check.status is not CheckStatus.REJECTED:
            self._pending[rec.req_id] = (rec, check)
        return check

    async def await_approval(self, req_id: int) -> Decision:
        if req_id not in self._pending:
            raise KeyError(f"no checked request {req_id}")
        if self._progress is not None:
            self._progress.at("approve")
        await self._pause()
        return Decision.APPROVED

    async def await_completion(self, req_id: int) -> Completion:
        rec, check = self._pending.pop(req_id)
        metres = self._rig.metres_drilled(rec.hole_idx, check.depth)
        if self._progress is not None:
            self._progress.at("drill", depth_m=0.0, target_m=metres)
            step = self._mps * DRILL_TICK_S
            while self._progress.depth_m < metres:
                await asyncio.sleep(DRILL_TICK_S)
                self._progress.depth_m = min(metres, self._progress.depth_m + step)
        self._lab.release(rec.hole_idx, metres)
        return Completion(req_id, rec.hole_idx, metres)
