"""Live drilling view: `geoopt live`. A read-only web page that animates the loop as it runs.

It reads exactly what an operator-side display could read, and nothing else:
- the OPC UA variables (recommendation, check, approval, dispatch, rig depth, KPIs), polled from
  the server EAE (or the emulator) hosts; and
- the run folder the planner writes (iteration records, model grid, released lab results).
It never imports the planner, the model or the simulators, so it keeps working unchanged when
real EAE devices replace the emulator and rig simulator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from geoopt.bridge.opcua_client import connect
from geoopt.config import Config
from geoopt.data.prepare import prepared_paths

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
SECTION_HALF_WIDTH_M = 450.0
SECTION_BAND_M = 40.0  # samples and traces within this of the section plane are drawn


@dataclass
class LiveState:
    cfg: Config
    opc: dict[str, float] = field(default_factory=dict)
    opc_ok: bool = False


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        return _clean(v.item())
    return v


def _json(data: Any) -> JSONResponse:
    return JSONResponse(_clean(data))


async def poll_opcua(state: LiveState) -> None:
    """Keep a copy of every node's value; reconnect quietly while nothing is serving."""
    names = list(state.cfg.nodes.nodes)
    while True:
        try:
            async with connect(state.cfg) as s:
                state.opc_ok = True
                while True:
                    state.opc = {n: float(await s.read(n)) for n in names}
                    await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except Exception:  # server not up yet, or went away
            state.opc_ok = False
            await asyncio.sleep(1.0)


# Run folder ------------------------------------------------------------------------------


def latest_run(runs_dir: Path) -> Path | None:
    runs = [p for p in runs_dir.glob("*") if (p / "config.yaml").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _records(run: Path) -> list[dict]:
    out = []
    for p in sorted(run.glob("iter_*/record.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            out.append(json.loads(p.read_text()))
    return out


def _lab_dir(run: Path) -> Path:
    if (run / "lab").exists():
        return run / "lab"
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    return Path(cfg["loop"]["live_lab_dir"])


def _lab(run: Path, hole_idx: int) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(_lab_dir(run) / f"{hole_idx}.parquet")
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=64)
def _grid(path: str, mtime: float) -> tuple[RegularGridInterpolator, dict] | None:
    g = pd.read_parquet(path)
    axes = [np.sort(g[c].unique()) for c in ("x", "y", "z")]
    g = g.sort_values(["x", "y", "z"])
    vol = g["p_ore"].to_numpy().reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    interp = RegularGridInterpolator(axes, vol, bounds_error=False, fill_value=0.0)
    plan = vol.max(axis=2)  # x by y: highest P(ore) at any depth
    ext = {"x": [axes[0][0], axes[0][-1]], "y": [axes[1][0], axes[1][-1]],
           "z": [axes[2][0], axes[2][-1]], "plan": plan.T.round(3).tolist()}  # fmt: skip
    return interp, ext


def _grid_for(run: Path, iteration: int | None) -> tuple[RegularGridInterpolator, dict] | None:
    iters = sorted(int(p.parent.name[5:]) for p in run.glob("iter_*/grid.parquet"))
    if not iters:
        return None
    it = max((i for i in iters if iteration is None or i <= iteration), default=iters[0])
    path = run / f"iter_{it:03d}" / "grid.parquet"
    try:
        return _grid(str(path), path.stat().st_mtime)
    except (OSError, ValueError):
        return None


# App ---------------------------------------------------------------------------------------


def build_app(cfg: Config) -> Starlette:
    state = LiveState(cfg)
    paths = prepared_paths(cfg.data.processed_dir)
    cand = pd.read_parquet(paths.candidates).set_index("hole_idx", drop=False)
    seed = pd.read_parquet(paths.seed_composites)
    collars = pd.read_parquet(paths.collars)
    seed_collars = collars[collars["hole_id"].isin(set(seed["hole_id"]))]
    baseline = None
    if cfg.kpi.baseline_file and Path(cfg.kpi.baseline_file).exists():
        b = pd.read_parquet(cfg.kpi.baseline_file)
        baseline = {"metres": b["metres"].round(1).tolist(), "blocks": b["new_ore_blocks"].tolist()}

    def run_dir(request: Request) -> Path | None:
        name = request.query_params.get("run")
        return cfg.loop.runs_dir / name if name else latest_run(cfg.loop.runs_dir)

    async def index(_: Request) -> FileResponse:
        return FileResponse(STATIC / "index.html")

    async def static_info(_: Request) -> JSONResponse:
        lim = cfg.limits
        return _json(
            {
                "cutoff_pct": cfg.model.cutoff_pct,
                "drill_cost_aud_per_m": cfg.kpi.drill_cost_aud_per_m,
                "max_depth_m": lim.max_depth_m,
                "bounds": lim.site_bounds.model_dump() if lim.site_bounds else None,
                "no_go": [z.model_dump() for z in lim.no_go_zones],
                "candidates": cand[["hole_idx", "x", "y"]].round(1).to_numpy().tolist(),
                "seed_collars": seed_collars[["x", "y"]].round(1).to_numpy().tolist(),
                "baseline": baseline,
                "metres_per_second": cfg.rig.metres_per_second,
                "block_t": float(np.prod(cfg.backtest.block_size_m)) * cfg.kpi.density_t_m3,
                "density_t_m3": cfg.kpi.density_t_m3,
            }
        )

    async def live_state(request: Request) -> JSONResponse:
        run = run_dir(request)
        recs = _records(run) if run else []
        rows = []
        for r in recs:
            h = cand.loc[r["hole_idx"]] if r["hole_idx"] in cand.index else None
            lab = _lab(run, r["hole_idx"]) if r.get("metres_drilled", 0) > 0 else None
            ore = float(lab.loc[lab["cu"] > cfg.model.cutoff_pct, "length_m"].sum()) if (
                lab is not None and len(lab)
            ) else 0.0  # fmt: skip
            rows.append(
                {
                    "iteration": r["iteration"],
                    "hole_idx": r["hole_idx"],
                    "hole_id": r["hole_id"],
                    **(
                        {}
                        if h is None
                        else {k: float(h[k]) for k in ("x", "y", "z", "azimuth", "dip", "depth")}
                    ),
                    "check": r["check_status"],
                    "decision": r.get("decision"),
                    "metres": r.get("metres_drilled", 0.0),
                    "ore_m": ore,
                    "prob_ore": r.get("prob_ore"),
                    "exp_ore_m": r.get("exp_ore_m"),
                    "kpis": r.get("kpis") or {},
                }
            )
        opc = state.opc if state.opc_ok else None
        current = None
        hole_idx = int(opc["Rec.HoleIdx"]) if opc and opc.get("Rec.ReqId", 0) > 0 else None
        if hole_idx is None and rows:
            hole_idx = rows[-1]["hole_idx"]
        if hole_idx is not None and hole_idx in cand.index:
            h = cand.loc[hole_idx]
            current = {k: float(h[k]) for k in ("x", "y", "z", "azimuth", "dip", "depth")}
            current |= {"hole_idx": hole_idx, "hole_id": str(h["hole_id"])}
        return _json(
            {
                "run": run.name if run else None,
                "records": rows,
                "opc": opc,
                "current": current,
            }
        )

    async def section(request: Request) -> JSONResponse:
        run = run_dir(request)
        q = request.query_params
        hole_idx = int(q["hole"])
        it = int(q["iter"]) if q.get("iter") else None
        if run is None or hole_idx not in cand.index:
            return _json({"ok": False})
        h = cand.loc[hole_idx]
        az = math.radians(float(h["azimuth"]))
        if float(h["dip"]) <= -89.0:
            az = math.pi / 2  # vertical hole: look along east-west
        u = np.array([math.sin(az), math.cos(az)])  # along-section unit vector (x, y)
        c = np.array([float(h["x"]), float(h["y"])])
        grid = _grid_for(run, it)
        s_axis = np.linspace(-SECTION_HALF_WIDTH_M, SECTION_HALF_WIDTH_M, 181)
        out: dict[str, Any] = {"ok": True, "s": [s_axis[0], s_axis[-1]], "hole_idx": hole_idx}
        if grid is not None:
            interp, ext = grid
            z_axis = np.linspace(ext["z"][1], ext["z"][0], 72)  # top to bottom
            ss, zz = np.meshgrid(s_axis, z_axis)
            pts = np.column_stack([c[0] + ss.ravel() * u[0], c[1] + ss.ravel() * u[1], zz.ravel()])
            out["z"] = [z_axis[0], z_axis[-1]]
            out["p"] = interp(pts).reshape(ss.shape).round(3).tolist()
            out["plan"] = {k: ext[k] for k in ("x", "y", "plan")}

        def project(df: pd.DataFrame) -> pd.DataFrame:
            d = df[["x", "y"]].to_numpy() - c
            s = d @ u
            off = np.abs(d @ np.array([-u[1], u[0]]))
            keep = (off <= SECTION_BAND_M) & (np.abs(s) <= SECTION_HALF_WIDTH_M)
            return df.assign(s=s)[keep]

        upto = int(q["upto"]) if q.get("upto") else None  # replay: only holes drilled by then
        drilled = [
            r["hole_idx"]
            for r in _records(run)
            if r.get("metres_drilled", 0) > 0 and (upto is None or r["iteration"] <= upto)
        ]
        labs = [x for x in (_lab(run, i) for i in drilled if i != hole_idx) if x is not None]
        known = pd.concat([seed, *labs], ignore_index=True) if labs else seed
        near = project(known)
        out["samples"] = near[["s", "z", "cu"]].round(3).to_numpy().tolist()
        done = any(
            r["hole_idx"] == hole_idx and r.get("metres_drilled", 0) > 0 for r in _records(run)
        )
        lab = _lab(run, hole_idx) if done else None
        out["assays"] = (
            lab[["from_m", "to_m", "cu"]].round(3).to_numpy().tolist() if lab is not None else None
        )
        out["it"] = it
        return _json(out)

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        task = asyncio.create_task(poll_opcua(state))
        try:
            yield
        finally:
            task.cancel()

    routes = [
        Route("/", index),
        Route("/api/static", static_info),
        Route("/api/state", live_state),
        Route("/api/section", section),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
