"""Live drilling view: `geoopt demo` (run + view) and `geoopt live` (replay finished runs).

A read-only web page that animates the loop. It reads only:
- the run folder the planner writes (iteration records, model grid, released lab results); and
- during `geoopt demo`, the `Progress` the loop updates (cycle stage, rig depth).
It never imports the planner, the model or the simulators; `geoopt demo` hands it the run as an
opaque coroutine to start alongside the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from geoopt.config import Config
from geoopt.data.prepare import prepared_paths
from geoopt.progress import STAGES, Progress

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
SECTION_HALF_WIDTH_M = 450.0
SECTION_BAND_M = 40.0  # samples and traces within this of the section plane are drawn


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


def _lab(run: Path, hole_idx: int) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(run / "lab" / f"{hole_idx}.parquet")
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


def _report(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("demo run failed", exc_info=task.exception())


# App ---------------------------------------------------------------------------------------


def build_app(
    cfg: Config,
    progress: Progress | None = None,
    job: Callable[[], Awaitable[None]] | None = None,
) -> Starlette:
    """`job` (the demo run) starts with the server; `progress` is what it reports."""
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
                "stages": list(STAGES),
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
            ore_rows = lab[lab["cu"] > cfg.model.cutoff_pct] if lab is not None else None
            ore = float(ore_rows["length_m"].sum()) if ore_rows is not None else 0.0
            mean_cu = (  # length-weighted Cu over the ore intervals
                float((ore_rows["cu"] * ore_rows["length_m"]).sum() / ore) if ore > 0 else None
            )
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
                    "mean_cu": mean_cu,
                    "prob_ore": r.get("prob_ore"),
                    "exp_ore_m": r.get("exp_ore_m"),
                    "kpis": r.get("kpis") or {},
                }
            )
        live = progress.as_dict() if progress is not None and progress.req_id > 0 else None
        current = None
        hole_idx = live["hole_idx"] if live else None
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
                "live": live,
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
        task = asyncio.create_task(job()) if job is not None else None
        if task is not None:
            task.add_done_callback(_report)
        try:
            yield
        finally:
            if task is not None:
                task.cancel()

    routes = [
        Route("/", index),
        Route("/api/static", static_info),
        Route("/api/state", live_state),
        Route("/api/section", section),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
