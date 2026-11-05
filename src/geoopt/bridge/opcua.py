"""OPC UA bridge: the planner side of the EAE contract. Same code for the emulator and EAE."""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from geoopt.bridge.opcua_client import (
    OpcUaSession,
    await_approval,
    await_check,
    connect,
    next_req_id,
)
from geoopt.config import Config
from geoopt.protocol import CheckResult, CheckStatus, Completion, Decision, RecommendationMsg

log = logging.getLogger(__name__)

# Payload written before the trigger (Rec.ReqId), in this order.
_REC_FIELDS = {
    "Rec.HoleIdx": "hole_idx",
    "Rec.CollarX": "collar_x",
    "Rec.CollarY": "collar_y",
    "Rec.CollarZ": "collar_z",
    "Rec.Azimuth": "azimuth",
    "Rec.Dip": "dip",
    "Rec.Depth": "depth",
    "Rec.ProbOre": "prob_ore",
    "Rec.ExpOreM": "exp_ore_m",
    "Rec.Score": "score",
}


class OpcUaBridge:
    def __init__(self, session: OpcUaSession, cfg: Config) -> None:
        self._s = session
        self._cfg = cfg
        self._hole_of: dict[int, int] = {}

    @classmethod
    @asynccontextmanager
    async def connect(cls, cfg: Config) -> AsyncIterator[OpcUaBridge]:
        async with connect(cfg) as session:
            yield cls(session, cfg)

    async def next_req_id(self) -> int:
        return max(await next_req_id(self._s), await self._s.read("Done.ReqId") + 1)

    async def submit(self, rec: RecommendationMsg) -> CheckResult:
        for node, attr in _REC_FIELDS.items():
            await self._s.write(node, getattr(rec, attr))
        await self._s.write("Rec.ReqId", rec.req_id)
        log.info("recommendation sent req_id=%s hole_idx=%s", rec.req_id, rec.hole_idx)
        self._hole_of[rec.req_id] = rec.hole_idx
        return await await_check(self._s, rec.req_id, self._cfg.opcua.timeouts.check_s)

    async def await_approval(self, req_id: int) -> Decision:
        return await await_approval(self._s, req_id, self._cfg.opcua.timeouts.approval_s)

    async def publish_kpis(self, kpis: dict[str, float]) -> None:
        for node, key in _KPI_NODES.items():
            if node not in self._cfg.nodes.nodes:
                continue
            v = kpis.get(key, float("nan"))
            v = -1.0 if not math.isfinite(v) else v
            await self._s.write(node, int(v) if node == "Kpi.Holes" else float(v))

    async def await_completion(self, req_id: int) -> Completion:
        await self._s.wait_for_req_id("Done.ReqId", req_id, self._cfg.opcua.timeouts.drill_s)
        metres = float(await self._s.read("Done.MetresDrilled"))
        log.info("completion req_id=%s metres=%.1f", req_id, metres)
        return Completion(req_id, self._hole_of.pop(req_id), metres)


# KPI field -> node. NaN (no baseline yet) is written as -1 so integer-minded HMIs can hide it.
_KPI_NODES = {
    "Kpi.Holes": "holes",
    "Kpi.Metres": "metres",
    "Kpi.OreMetres": "ore_m",
    "Kpi.NewOreMt": "new_ore_mt",
    "Kpi.NewCuKt": "new_cu_kt",
    "Kpi.DrillCostAudM": "drill_cost_aud_m",
    "Kpi.BaselineMetres": "baseline_metres",
    "Kpi.MetresSaved": "metres_saved",
    "Kpi.CostSavedAudM": "cost_saved_aud_m",
}


@dataclass(frozen=True)
class RoundTripResult:
    req_id: int
    check: CheckResult
    decision: Decision | None  # None when the check rejected the request


async def roundtrip(cfg: Config, depth: float, hole_idx: int = 0) -> RoundTripResult:
    """Phase 0 spike: one dummy recommendation -> check -> approval (no rig needed).

    The collar sits at the centre of the site bounds with a -60 degree dip, so only depth
    decides between ACCEPTED and CLIPPED.
    """
    b = cfg.limits.site_bounds
    x, y = ((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2) if b else (0.0, 0.0)
    async with OpcUaBridge.connect(cfg) as bridge:
        req_id = await bridge.next_req_id()
        rec = RecommendationMsg(
            req_id=req_id,
            hole_idx=hole_idx,
            collar_x=x,
            collar_y=y,
            collar_z=0.0,
            azimuth=90.0,
            dip=-60.0,
            depth=depth,
            prob_ore=0.0,
            exp_ore_m=0.0,
            score=0.0,
        )
        check = await bridge.submit(rec)
        if check.status is CheckStatus.REJECTED:
            return RoundTripResult(req_id, check, None)
        return RoundTripResult(req_id, check, await bridge.await_approval(req_id))
