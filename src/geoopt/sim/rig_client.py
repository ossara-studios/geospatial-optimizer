"""Rig + lab simulator as an OPC UA client (Plan A topology: EAE hosts the server).

Watches Disp.ReqId, "drills" the hole at a configurable demo speed while publishing
SYNTHETIC telemetry, then releases the lab results and reports Rig.State = COMPLETE.
"""

from __future__ import annotations

import asyncio
import logging

from geoopt.bridge.opcua_client import OpcUaSession, connect
from geoopt.config import Config
from geoopt.protocol import RigState
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim

log = logging.getLogger(__name__)


class RigClient:
    def __init__(self, session: OpcUaSession, cfg: Config, rig: RigSim, lab: LabSim) -> None:
        self._s = session
        self._cfg = cfg
        self._rig = rig
        self._lab = lab
        self._last = 0

    async def start(self) -> None:
        # Never re-drill a request that was dispatched before this client started.
        self._last = int(await self._s.read("Disp.ReqId"))
        await self._s.write("Rig.State", int(RigState.IDLE))

    async def process_once(self) -> bool:
        req_id = int(await self._s.read("Disp.ReqId"))
        if req_id <= self._last:
            return False
        self._last = req_id
        hole_idx = int(await self._s.read("Disp.HoleIdx"))
        target = self._rig.metres_drilled(hole_idx, float(await self._s.read("Disp.Depth")))
        log.info("rig drilling req_id=%s hole_idx=%s target=%.1f m", req_id, hole_idx, target)
        await self._s.write("Rig.ReqId", req_id)
        await self._s.write("Rig.State", int(RigState.DRILLING))

        step_s = self._cfg.rig.telemetry_period_s
        step_m = self._cfg.rig.metres_per_second * step_s
        depth = 0.0
        while depth < target:
            depth = min(target, depth + step_m)
            t = self._rig.telemetry(depth)
            await self._s.write("Rig.DepthDrilled", depth)
            await self._s.write("Rig.Torque", t.torque_knm)
            await self._s.write("Rig.WaterLpm", t.water_lpm)
            await self._s.write("Rig.PowerKw", t.power_kw)
            if depth < target:
                await asyncio.sleep(step_s)

        self._lab.release(hole_idx, target)  # lab results land before completion is reported
        await self._s.write("Rig.State", int(RigState.COMPLETE))
        log.info("rig complete req_id=%s metres=%.1f", req_id, target)
        return True

    async def run(self) -> None:
        await self.start()
        while True:
            await self.process_once()
            await asyncio.sleep(self._cfg.opcua.poll_interval_s)


async def serve_rig(cfg: Config, rig: RigSim, lab: LabSim) -> None:
    async with connect(cfg) as session:
        await RigClient(session, cfg, rig, lab).run()
