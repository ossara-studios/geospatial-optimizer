"""EAE emulator: an OPC UA server with the same node map and limit logic as the EAE project.

It plays the OT side so the Python loop can be built and tested without EAE. Like a PLC
scan, it polls `Rec.ReqId` and acts once per new, higher ReqId.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from asyncua import Node, Server, ua

from geoopt.bridge.opcua_client import VARIANT_TYPES
from geoopt.config import Config
from geoopt.limits import check_recommendation
from geoopt.protocol import CheckResult, CheckStatus, Decision, RigState

log = logging.getLogger(__name__)

Approver = Callable[[int, CheckResult], Awaitable[Decision]]


async def auto_approve(req_id: int, result: CheckResult) -> Decision:
    return Decision.APPROVED


async def cli_approve(req_id: int, result: CheckResult) -> Decision:
    prompt = (
        f"\nReqId={req_id} status={result.status.name} depth={result.depth:.1f} m"
        " -> approve? [y/n]: "
    )
    while True:
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
        if answer in ("y", "yes"):
            return Decision.APPROVED
        if answer in ("n", "no"):
            return Decision.DENIED


class Emulator:
    def __init__(self, cfg: Config, approver: Approver) -> None:
        self._cfg = cfg
        self._approver = approver
        self._server = Server()
        self._nodes: dict[str, tuple[Node, ua.VariantType]] = {}
        self._last_req_id = 0
        self._last_stale: int | None = None
        self._dispatched: int | None = None  # ReqId sent to the rig, awaiting completion

    async def start(self) -> None:
        await self._server.init()
        self._server.set_endpoint(self._cfg.opcua.endpoint)
        self._server.set_server_name("geoopt EAE emulator")
        self._server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        idx = await self._server.register_namespace(self._cfg.nodes.namespace_uri)
        folder = await self._server.nodes.objects.add_object(idx, "GeoOpt")
        for name, spec in self._cfg.nodes.nodes.items():
            if spec.id is None:
                raise ValueError(f"emulator needs `id` (not `node_id`) for node {name}")
            vt = VARIANT_TYPES[spec.type]
            initial = 0.0 if vt is ua.VariantType.Double else 0
            node = await folder.add_variable(ua.NodeId(spec.id, idx), name, initial, vt)
            await node.set_writable()
            self._nodes[name] = (node, vt)
        await self._server.start()
        log.info("emulator listening endpoint=%s", self._cfg.opcua.endpoint)

    async def stop(self) -> None:
        await self._server.stop()

    async def _read(self, name: str) -> Any:
        return await self._nodes[name][0].read_value()

    async def _write(self, name: str, value: Any) -> None:
        node, vt = self._nodes[name]
        await node.write_value(ua.Variant(value, vt))

    async def process_once(self) -> bool:
        """One scan: forward a rig completion, then handle a new recommendation, if any."""
        await self._forward_completion()
        req_id = await self._read("Rec.ReqId")
        if req_id == self._last_req_id:
            return False
        if req_id < self._last_req_id:
            if req_id != self._last_stale:
                log.warning("stale req_id=%s ignored (last=%s)", req_id, self._last_req_id)
                self._last_stale = req_id
            return False
        self._last_req_id = req_id

        result = check_recommendation(
            x=await self._read("Rec.CollarX"),
            y=await self._read("Rec.CollarY"),
            azimuth=await self._read("Rec.Azimuth"),
            dip=await self._read("Rec.Dip"),
            depth=await self._read("Rec.Depth"),
            limits=self._cfg.limits,
        )
        await self._write("Chk.Status", int(result.status))
        await self._write("Chk.Reason", int(result.reason))
        await self._write("Chk.Depth", result.depth)
        await self._write("Chk.ReqId", req_id)
        log.info(
            "checked req_id=%s status=%s reason=%s depth=%.1f",
            req_id,
            result.status.name,
            result.reason.name,
            result.depth,
        )
        if result.status is CheckStatus.REJECTED:
            return True

        decision = await self._approver(req_id, result)
        await self._write("Appr.Decision", int(decision))
        await self._write("Appr.ReqId", req_id)
        log.info("approval req_id=%s decision=%s", req_id, decision.name)
        if decision is Decision.APPROVED:
            await self._write("Disp.HoleIdx", await self._read("Rec.HoleIdx"))
            await self._write("Disp.Depth", result.depth)
            await self._write("Disp.ReqId", req_id)
            self._dispatched = req_id
            log.info("dispatched req_id=%s depth=%.1f", req_id, result.depth)
        return True

    async def _forward_completion(self) -> None:
        if self._dispatched is None:
            return
        state = await self._read("Rig.State")
        if state == RigState.FAULT:
            log.error("rig fault on req_id=%s", self._dispatched)
            return
        if state != RigState.COMPLETE or await self._read("Rig.ReqId") != self._dispatched:
            return
        await self._write("Done.MetresDrilled", await self._read("Rig.DepthDrilled"))
        await self._write("Done.ReqId", self._dispatched)
        log.info("completion forwarded req_id=%s", self._dispatched)
        self._dispatched = None

    async def run(self) -> None:
        while True:
            await self.process_once()
            await asyncio.sleep(self._cfg.opcua.poll_interval_s)


async def serve(cfg: Config, approver: Approver) -> None:
    emulator = Emulator(cfg, approver)
    await emulator.start()
    try:
        await emulator.run()
    finally:
        await emulator.stop()
