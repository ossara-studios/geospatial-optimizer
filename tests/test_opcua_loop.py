"""Full OPC UA path on localhost: emulator (EAE stand-in), rig-sim client and planner loop."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from geoopt.bridge.opcua import OpcUaBridge
from geoopt.bridge.opcua_client import connect
from geoopt.config import Config, Limits
from geoopt.data.prepare import prepare, prepared_paths
from geoopt.emulator.server import Emulator, auto_approve
from geoopt.limits import check_recommendation
from geoopt.protocol import Reason, RecommendationMsg
from geoopt.run import run_opcua
from geoopt.sim.lab import LabSim
from geoopt.sim.rig import RigSim
from geoopt.sim.rig_client import RigClient

pytestmark = pytest.mark.opcua

LIMITS = Limits(
    max_depth_m=500.0,
    site_bounds={"x_min": 0, "x_max": 1000, "y_min": 0, "y_max": 1000},
    no_go_zones=[
        {"kind": "rect", "name": "drainage", "x_min": 100, "x_max": 200, "y_min": 0, "y_max": 1000},
        {"kind": "circle", "name": "heritage", "x": 800, "y": 800, "radius": 50},
    ],
    dip_range=(-90.0, -45.0),
    azimuth_range=(0.0, 360.0),
)

CASES = [  # (x, y, azimuth, dip, depth): one per reason code, plus a clip
    (500, 500, 90, -60, 300),
    (500, 500, 90, -60, 650),
    (1500, 500, 90, -60, 300),
    (150, 500, 90, -60, 300),
    (820, 810, 90, -60, 300),
    (500, 500, 90, -30, 300),
    (500, 500, 400, -60, 300),
]


async def test_emulator_matches_inprocess_for_every_reason_code(local_cfg: Config) -> None:
    cfg = local_cfg.model_copy(update={"limits": LIMITS})
    emulator = Emulator(cfg, auto_approve)
    await emulator.start()
    task = asyncio.create_task(emulator.run())
    try:
        async with OpcUaBridge.connect(cfg) as bridge:
            req_id = await bridge.next_req_id()
            seen = set()
            for i, (x, y, az, dip, depth) in enumerate(CASES):
                msg = RecommendationMsg(
                    req_id=req_id + i,
                    hole_idx=i,
                    collar_x=x,
                    collar_y=y,
                    collar_z=0,
                    azimuth=az,
                    dip=dip,
                    depth=depth,
                    prob_ore=0.5,
                    exp_ore_m=10,
                    score=1,
                )
                over_ua = await bridge.submit(msg)
                assert over_ua == check_recommendation(x, y, az, dip, depth, LIMITS)
                seen.add(over_ua.reason)
            assert seen == set(Reason)
    finally:
        task.cancel()
        await emulator.stop()


async def test_three_iteration_loop_over_opcua(
    prepared_cfg: Config, local_cfg: Config, tmp_path: Path
) -> None:
    cfg = prepared_cfg.model_copy(
        update={
            "opcua": local_cfg.opcua,
            "loop": prepared_cfg.loop.model_copy(
                update={"max_holes": 3, "live_lab_dir": tmp_path / "live_lab"}
            ),
            "rig": prepared_cfg.rig.model_copy(
                update={"metres_per_second": 5000.0, "telemetry_period_s": 0.02}
            ),
        }
    )
    prepare(cfg)
    paths = prepared_paths(cfg.data.processed_dir)
    emulator = Emulator(cfg, auto_approve)
    await emulator.start()
    tasks = [asyncio.create_task(emulator.run())]
    try:
        async with connect(cfg) as rig_session:
            rig = RigClient(
                rig_session,
                cfg,
                RigSim(pd.read_parquet(paths.candidates), cfg.seed),
                LabSim(paths.heldout_composites, cfg.loop.live_lab_dir),
            )
            await rig.start()

            async def rig_loop() -> None:
                while True:
                    await rig.process_once()
                    await asyncio.sleep(cfg.opcua.poll_interval_s)

            tasks.append(asyncio.create_task(rig_loop()))
            run_dir = tmp_path / "run"
            records = await asyncio.wait_for(run_opcua(cfg, "adaptive", run_dir), timeout=60)
    finally:
        for t in tasks:
            t.cancel()
        await emulator.stop()

    drilled = [r for r in records if r.metres_drilled > 0]
    assert len(drilled) == 3
    assert [r.req_id for r in records] == list(range(1, len(records) + 1))
    lab_files = {int(p.stem) for p in cfg.loop.live_lab_dir.glob("*.parquet")}
    assert lab_files == {r.hole_idx for r in drilled}
    assert (run_dir / "iter_003" / "record.json").exists()
