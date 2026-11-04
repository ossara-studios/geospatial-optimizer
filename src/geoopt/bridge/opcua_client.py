"""Minimal async OPC UA client helpers for the planner side.

Works unchanged against the EAE emulator and real EAE: only the endpoint URL and
the node map in config change.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from asyncua import Client, Node, ua

from geoopt.config import Config, NodeMap, OpcUaConfig
from geoopt.protocol import CheckResult, CheckStatus, Decision, Reason

log = logging.getLogger(__name__)

VARIANT_TYPES: dict[str, ua.VariantType] = {
    "UInt32": ua.VariantType.UInt32,
    "Int16": ua.VariantType.Int16,
    "Double": ua.VariantType.Double,
}


class OpcUaSession:
    """Named read/write/wait access to the nodes in the node map."""

    def __init__(self, nodes: dict[str, tuple[Node, ua.VariantType]], cfg: OpcUaConfig) -> None:
        self._nodes = nodes
        self._cfg = cfg

    async def read(self, name: str) -> Any:
        node, _ = self._nodes[name]
        return await node.read_value()

    async def write(self, name: str, value: Any) -> None:
        node, vt = self._nodes[name]
        # No timestamps: many PLC-hosted servers reject writes that carry them.
        await node.write_value(ua.DataValue(ua.Variant(value, vt)))

    async def wait_for_req_id(self, name: str, req_id: int, timeout_s: float | None) -> None:
        """Poll `name` until it echoes `req_id`. Raises TimeoutError after `timeout_s`."""
        start = time.monotonic()
        last_log = start
        last_seen: int | None = None
        while True:
            seen = await self.read(name)
            if seen == req_id:
                return
            if seen != last_seen and seen > req_id:
                log.warning("unexpected echo node=%s seen=%s expected=%s", name, seen, req_id)
            last_seen = seen
            now = time.monotonic()
            if timeout_s is not None and now - start > timeout_s:
                raise TimeoutError(f"{name} did not echo ReqId={req_id} within {timeout_s}s")
            if now - last_log >= self._cfg.log_wait_every_s:
                log.info("waiting node=%s req_id=%s elapsed_s=%.0f", name, req_id, now - start)
                last_log = now
            await asyncio.sleep(self._cfg.poll_interval_s)


async def _resolve_nodes(
    client: Client, node_map: NodeMap
) -> dict[str, tuple[Node, ua.VariantType]]:
    ns_idx: int | None = None
    resolved: dict[str, tuple[Node, ua.VariantType]] = {}
    for name, spec in node_map.nodes.items():
        if spec.node_id is not None:
            node_id = ua.NodeId.from_string(spec.node_id)
        else:
            if ns_idx is None:
                ns_idx = await client.get_namespace_index(node_map.namespace_uri)
            node_id = ua.NodeId(spec.id, ns_idx)
        resolved[name] = (client.get_node(node_id), VARIANT_TYPES[spec.type])
    return resolved


@asynccontextmanager
async def connect(cfg: Config) -> AsyncIterator[OpcUaSession]:
    client = Client(cfg.opcua.endpoint, timeout=cfg.opcua.timeouts.connect_s)
    log.info("connecting endpoint=%s security=%s", cfg.opcua.endpoint, cfg.opcua.security)
    async with client:
        nodes = await _resolve_nodes(client, cfg.nodes)
        yield OpcUaSession(nodes, cfg.opcua)


async def next_req_id(session: OpcUaSession) -> int:
    """One above any ReqId already on the server, so restarts never reuse an ID."""
    seen = [await session.read(n) for n in ("Rec.ReqId", "Chk.ReqId", "Appr.ReqId")]
    return max(seen) + 1


async def send_recommendation(session: OpcUaSession, req_id: int, depth: float) -> None:
    # Payload first; ReqId is the trigger and goes last, as its own write call.
    await session.write("Rec.Depth", depth)
    await session.write("Rec.ReqId", req_id)
    log.info("recommendation sent req_id=%s depth=%.1f", req_id, depth)


async def await_check(session: OpcUaSession, req_id: int, timeout_s: float) -> CheckResult:
    await session.wait_for_req_id("Chk.ReqId", req_id, timeout_s)
    result = CheckResult(
        status=CheckStatus(await session.read("Chk.Status")),
        reason=Reason(await session.read("Chk.Reason")),
        depth=await session.read("Chk.Depth"),
    )
    log.info(
        "check req_id=%s status=%s reason=%s depth=%.1f",
        req_id,
        result.status.name,
        result.reason.name,
        result.depth,
    )
    return result


async def await_approval(session: OpcUaSession, req_id: int, timeout_s: float | None) -> Decision:
    await session.wait_for_req_id("Appr.ReqId", req_id, timeout_s)
    decision = Decision(await session.read("Appr.Decision"))
    log.info("approval req_id=%s decision=%s", req_id, decision.name)
    return decision


@dataclass(frozen=True)
class RoundTripResult:
    req_id: int
    check: CheckResult
    decision: Decision | None  # None when the check rejected the request


async def roundtrip(cfg: Config, depth: float) -> RoundTripResult:
    """Send one recommendation and wait for the check and (if not rejected) the approval."""
    async with connect(cfg) as session:
        req_id = await next_req_id(session)
        await send_recommendation(session, req_id, depth)
        check = await await_check(session, req_id, cfg.opcua.timeouts.check_s)
        if check.status is CheckStatus.REJECTED:
            return RoundTripResult(req_id, check, None)
        decision = await await_approval(session, req_id, cfg.opcua.timeouts.approval_s)
        return RoundTripResult(req_id, check, decision)
