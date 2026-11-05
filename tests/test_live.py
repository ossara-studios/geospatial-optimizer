"""Live view API: reads only the run folder (and OPC UA when it's up)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from starlette.requests import Request

import geoopt
from geoopt.config import Config
from geoopt.data.prepare import prepare
from geoopt.live.server import build_app
from geoopt.run import run_inprocess

LIVE = Path(geoopt.__file__).parent / "live" / "server.py"


def test_live_view_never_imports_planner_model_or_simulators() -> None:
    src = LIVE.read_text(encoding="utf-8")
    assert not re.search(r"geoopt\.(planner|model|sim|loop|run)\b", src)
    assert not re.search(r"heldout|truth", src, re.IGNORECASE)


async def _get(app, path: str, query: str = "") -> dict:
    route = next(r for r in app.routes if r.path == path)
    scope = {"type": "http", "method": "GET", "path": path, "query_string": query.encode(),
             "headers": []}  # fmt: skip
    resp = await route.endpoint(Request(scope))
    return json.loads(resp.body)


async def test_state_and_section_from_a_finished_run(prepared_cfg: Config) -> None:
    prepare(prepared_cfg)
    await run_inprocess(prepared_cfg, "adaptive")
    app = build_app(prepared_cfg)
    static = await _get(app, "/api/static")
    assert static["cutoff_pct"] == prepared_cfg.model.cutoff_pct and static["candidates"]
    state = await _get(app, "/api/state")
    assert state["opc"] is None  # no OPC UA server in tests
    recs = state["records"]
    assert len(recs) == prepared_cfg.loop.max_holes
    assert recs[-1]["kpis"]["holes"] == prepared_cfg.loop.max_holes
    assert state["current"]["hole_idx"] == recs[-1]["hole_idx"]
    sec = await _get(app, "/api/section", f"hole={recs[0]['hole_idx']}&iter=1&upto=1")
    assert sec["ok"] and len(sec["p"]) > 0 and len(sec["p"][0]) == 181
    assert sec["assays"] is not None  # the hole was drilled, so its released assays are shown
