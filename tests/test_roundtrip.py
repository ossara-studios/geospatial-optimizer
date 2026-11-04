"""Round-trip against the emulator on localhost (the Phase 0 exit test)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from geoopt.bridge.opcua_client import connect, roundtrip, send_recommendation
from geoopt.config import Config
from geoopt.emulator.server import Approver, Emulator, auto_approve
from geoopt.protocol import CheckResult, CheckStatus, Decision, Reason

pytestmark = pytest.mark.opcua


async def _deny(req_id: int, result: CheckResult) -> Decision:
    return Decision.DENIED


async def _start(cfg: Config, approver: Approver, run: bool = True) -> AsyncIterator[Emulator]:
    emulator = Emulator(cfg, approver)
    await emulator.start()
    task = asyncio.create_task(emulator.run()) if run else None
    try:
        yield emulator
    finally:
        if task:
            task.cancel()
        await emulator.stop()


@pytest.fixture
async def approving(local_cfg: Config) -> AsyncIterator[Emulator]:
    async for emulator in _start(local_cfg, auto_approve):
        yield emulator


async def test_accepted_and_approved(local_cfg: Config, approving: Emulator) -> None:
    result = await roundtrip(local_cfg, depth=300.0)
    assert result.req_id == 1
    assert result.check == CheckResult(CheckStatus.ACCEPTED, Reason.NONE, 300.0)
    assert result.decision is Decision.APPROVED


async def test_clipped_and_req_id_increments(local_cfg: Config, approving: Emulator) -> None:
    first = await roundtrip(local_cfg, depth=300.0)
    second = await roundtrip(local_cfg, depth=650.0)
    assert second.req_id == first.req_id + 1
    assert second.check == CheckResult(CheckStatus.CLIPPED, Reason.DEPTH_CLIPPED, 500.0)
    assert second.decision is Decision.APPROVED


async def test_denied(local_cfg: Config) -> None:
    async for _ in _start(local_cfg, _deny):
        result = await roundtrip(local_cfg, depth=300.0)
        assert result.decision is Decision.DENIED


async def test_stale_req_id_is_ignored(local_cfg: Config, approving: Emulator) -> None:
    await roundtrip(local_cfg, depth=300.0)
    await roundtrip(local_cfg, depth=300.0)
    async with connect(local_cfg) as session:
        await send_recommendation(session, req_id=1, depth=999.0)
        await asyncio.sleep(0.3)
        assert await session.read("Chk.ReqId") == 2
        assert await session.read("Chk.Depth") == 300.0


async def test_check_timeout_when_nothing_answers(local_cfg: Config) -> None:
    cfg = local_cfg.model_copy(
        update={
            "opcua": local_cfg.opcua.model_copy(
                update={"timeouts": local_cfg.opcua.timeouts.model_copy(update={"check_s": 0.3})}
            )
        }
    )
    async for _ in _start(cfg, auto_approve, run=False):
        with pytest.raises(TimeoutError):
            await roundtrip(cfg, depth=300.0)
