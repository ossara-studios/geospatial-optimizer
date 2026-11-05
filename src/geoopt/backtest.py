"""Retrospective backtest: every strategy, same seed set, same candidates, same metre budget.

Evaluation sits outside the planner: it reads held-out truth through LabSim to score each
run, but never feeds it to the model or the strategy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.stats import norm

from geoopt.blocks import block_agreement, block_centres, expected_grade
from geoopt.config import AdaptiveVariant, Config, SeedSet
from geoopt.data.prepare import prepare, prepared_paths
from geoopt.logging_setup import configure_logging
from geoopt.loop import IterationRecord
from geoopt.metrics import classification, metres_to_target, on_metre_grid, ore_metres
from geoopt.model.gp import GPOreModel
from geoopt.run import build_inprocess_loop

log = logging.getLogger(__name__)

CURVE_METRICS = [
    "ore_m",
    "f1_undrilled",
    "bacc_undrilled",
    "f1_near",
    "bacc_near",
    "f1_undrilled_eg",
    "bacc_undrilled_eg",
    "f1_near_eg",
    "bacc_near_eg",
]
BLOCK_METRICS = ["block_f1", "block_bacc", "block_ore_volume_err", "block_metal_err"]
REFERENCE_FILE = "reference_blocks.parquet"


def _seed_set_cfg(cfg: Config, ss: SeedSet, out_dir: Path) -> Config:
    return cfg.model_copy(
        update={
            "split": cfg.split.model_copy(
                update={"method": ss.method, "n_seed": ss.n_seed, "seed": ss.seed}
            ),
            "data": cfg.data.model_copy(update={"processed_dir": out_dir / "data" / ss.name}),
            "loop": cfg.loop.model_copy(
                update={
                    "max_holes": cfg.backtest.max_holes,
                    "max_metres": cfg.backtest.metre_budget,
                }
            ),
        }
    )


def reference_blocks(cfg: Config, processed_dir: Path, out_dir: Path) -> Path:
    """Reference block model: GP fitted on every hole's assays (written once per out_dir).

    Blocks lie within block_radius_m of any assayed composite. Evaluation only: no strategy
    sees it. Each strategy's block estimate is scored against it at metre checkpoints.
    """
    path = out_dir / REFERENCE_FILE
    if path.exists():
        return path
    bt = cfg.backtest
    paths = prepared_paths(processed_dir)
    comps = pd.concat(
        [pd.read_parquet(paths.seed_composites), pd.read_parquet(paths.heldout_composites)]
    )
    xyz, cu = comps[["x", "y", "z"]].to_numpy(), comps["cu"].to_numpy()
    model = GPOreModel(cfg.model.model_copy(update={"max_train": bt.reference_max_train}), cfg.seed)
    model.fit(xyz, cu)
    blocks = block_centres(xyz, bt.block_size_m, bt.block_radius_m)
    mean, std = model.predict(blocks)
    grade = expected_grade(mean, std, cfg.model.log_eps)
    ref = pd.DataFrame({"x": blocks[:, 0], "y": blocks[:, 1], "z": blocks[:, 2], "grade": grade})
    ref.to_parquet(path, index=False)
    log.info(
        "reference blocks=%d ore_blocks=%d composites=%d",
        len(blocks),
        int((grade > cfg.model.cutoff_pct).sum()),
        len(comps),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


async def _one_run(
    cfg: Config,
    seed_set: str,
    strategy: str,
    rng_seed: int,
    run_dir: Path,
    label: str = "",
    ref_path: Path | None = None,
) -> pd.DataFrame:
    loop, lab = build_inprocess_loop(cfg, strategy, run_dir, rng_seed=rng_seed, write_grid=False)
    truth = lab.truth_for_scoring()
    seed_xyz = pd.read_parquet(prepared_paths(cfg.data.processed_dir).seed_composites)[
        ["x", "y", "z"]
    ].to_numpy()
    known = [seed_xyz]  # composites with assays the planner has seen
    xyz = truth[["x", "y", "z"]].to_numpy()
    actual = (truth["cu"] > cfg.model.cutoff_pct).to_numpy()
    cutoff, threshold = cfg.model.cutoff_pct, cfg.backtest.ore_threshold
    near_radius = cfg.backtest.near_radius_m
    log_cut, eps = np.log(cutoff + cfg.model.log_eps), cfg.model.log_eps
    ref = pd.read_parquet(ref_path) if ref_path is not None else None
    block_every = cfg.backtest.block_every_m
    rows: list[dict] = []
    found = {"ore_m": 0.0, "next_block_m": 0.0}

    def score(record: IterationRecord | None) -> None:
        if record is not None and record.metres_drilled > 0:
            released = pd.read_parquet(run_dir / "lab" / f"{record.hole_idx}.parquet")
            found["ore_m"] += ore_metres(released, cutoff)
            known.append(released[["x", "y", "z"]].to_numpy())
        undrilled = ~truth["hole_idx"].isin(loop.drilled_holes).to_numpy()
        mean, std = loop.model.predict(xyz, include_noise=True)
        predicted = norm.sf((log_cut - mean) / std) > threshold  # P(ore) > threshold
        predicted_eg = expected_grade(mean, std, eps) > cutoff  # estimated grade > cutoff
        und = classification(predicted[undrilled], actual[undrilled])
        und_eg = classification(predicted_eg[undrilled], actual[undrilled])
        dist, _ = cKDTree(np.vstack(known)).query(xyz, distance_upper_bound=near_radius)
        near = undrilled & np.isfinite(dist)
        nr = classification(predicted[near], actual[near])
        nr_eg = classification(predicted_eg[near], actual[near])
        metres = 0.0 if record is None else record.cumulative_metres
        blocks = dict.fromkeys(BLOCK_METRICS, float("nan"))
        if ref is not None and metres >= found["next_block_m"]:
            b_mean, b_std = loop.model.predict(ref[["x", "y", "z"]].to_numpy())
            blocks = block_agreement(
                expected_grade(b_mean, b_std, eps), ref["grade"].to_numpy(), cutoff
            )
            found["next_block_m"] = (np.floor(metres / block_every) + 1) * block_every
        rows.append(
            {
                "seed_set": seed_set,
                "strategy": label or strategy,
                "beta": cfg.planner.beta if strategy == "adaptive" else float("nan"),
                "gamma": cfg.planner.gamma if strategy == "adaptive" else float("nan"),
                "delta": cfg.planner.delta if strategy == "adaptive" else float("nan"),
                "ore_weight": cfg.planner.ore_weight if strategy == "adaptive" else float("nan"),
                "rng_seed": rng_seed,
                "iteration": 0 if record is None else record.iteration,
                "holes": len(loop.drilled_holes),
                "cum_metres": metres,
                "ore_m": found["ore_m"],
                "f1_undrilled": und["f1"],
                "bacc_undrilled": und["balanced_accuracy"],
                "f1_all": classification(predicted, actual)["f1"],
                "n_undrilled": und["n"],
                "f1_near": nr["f1"],
                "bacc_near": nr["balanced_accuracy"],
                "n_near": nr["n"],
                "ore_frac_near": float(actual[near].mean()) if near.any() else float("nan"),
                "f1_undrilled_eg": und_eg["f1"],
                "bacc_undrilled_eg": und_eg["balanced_accuracy"],
                "f1_near_eg": nr_eg["f1"],
                "bacc_near_eg": nr_eg["balanced_accuracy"],
            }
            | blocks
        )

    score(None)
    loop.on_iteration = score
    await loop.run()
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class _Job:
    tag: str
    cfg: Config
    seed_set: str
    strategy: str
    rng_seed: int
    run_dir: Path
    label: str
    ref_path: Path | None = None


def _run_job(job: _Job) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = asyncio.run(
        _one_run(
            job.cfg, job.seed_set, job.strategy, job.rng_seed, job.run_dir, job.label, job.ref_path
        )
    )
    last = df.iloc[-1]
    log.info(
        "%s seed_set=%s strategy=%s rng=%s holes=%d metres=%.0f ore_m=%.0f "
        "f1_undrilled=%.3f f1_undrilled_eg=%.3f block_f1=%.3f seconds=%.0f",
        job.tag,
        job.seed_set,
        job.label or job.strategy,
        job.rng_seed,
        last["holes"],
        last["cum_metres"],
        last["ore_m"],
        last["f1_undrilled"],
        last["f1_undrilled_eg"],
        df["block_f1"].dropna().iloc[-1] if df["block_f1"].notna().any() else float("nan"),
        time.perf_counter() - t0,
    )
    return df


def _init_worker(threads: int, log_level: str) -> None:
    torch.set_num_threads(threads)
    configure_logging(log_level)


def _execute(jobs: list[_Job], workers: int, log_level: str) -> list[pd.DataFrame]:
    """Run independent backtest runs, in `workers` processes when > 1 (results keep job order)."""
    if workers <= 1:
        return [_run_job(j) for j in jobs]
    threads = max(1, (os.cpu_count() or 1) // workers)
    log.info("running %d runs on %d workers, %d torch threads each", len(jobs), workers, threads)
    ctx = multiprocessing.get_context("spawn")  # CUDA can't be forked
    with ProcessPoolExecutor(
        workers, mp_context=ctx, initializer=_init_worker, initargs=(threads, log_level)
    ) as pool:
        return list(pool.map(_run_job, jobs))


def run_backtest(cfg: Config, out_dir: Path | None = None, workers: int = 1) -> pd.DataFrame:
    bt = cfg.backtest
    out_dir = out_dir or bt.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    ref_path = None
    for ss in bt.seed_sets:
        ss_cfg = _seed_set_cfg(cfg, ss, out_dir)
        prepare(ss_cfg)
        ref_path = ref_path or reference_blocks(cfg, ss_cfg.data.processed_dir, out_dir)
        for strategy in bt.strategies:
            seeds = bt.random_seeds if strategy == "random" else [cfg.seed]
            for rng_seed in seeds:
                run_dir = out_dir / "runs" / ss.name / f"{strategy}-{rng_seed}"
                jobs.append(
                    _Job("backtest", ss_cfg, ss.name, strategy, rng_seed, run_dir, "", ref_path)
                )
    frames = _execute(jobs, workers, cfg.log_level)
    results = pd.concat(frames, ignore_index=True)
    results.to_parquet(out_dir / "results.parquet", index=False)
    summary = summarize_backtest(results, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_backtest(results, cfg, out_dir / "backtest.html")
    return results


def run_beta_sweep(cfg: Config, out_dir: Path | None = None, workers: int = 1) -> pd.DataFrame:
    """Adaptive at each `backtest.beta_sweep` beta (gamma 0)."""
    variants = [AdaptiveVariant(beta=b) for b in cfg.backtest.beta_sweep]
    return _run_variants(cfg, variants, "beta_sweep", out_dir, workers=workers)


def run_boundary_sweep(cfg: Config, out_dir: Path | None = None, workers: int = 1) -> pd.DataFrame:
    """Adaptive at each `backtest.boundary_sweep` (beta, gamma); joins the beta sweep too."""
    variants = cfg.backtest.boundary_sweep
    return _run_variants(cfg, variants, "boundary_sweep", out_dir, ["beta_sweep"], workers)


def run_delineation_sweep(
    cfg: Config, out_dir: Path | None = None, workers: int = 1
) -> pd.DataFrame:
    """Adaptive with the targeted variance-reduction term (`backtest.delineation_sweep`)."""
    variants = cfg.backtest.delineation_sweep
    join = ["beta_sweep", "boundary_sweep"]
    return _run_variants(cfg, variants, "delineation_sweep", out_dir, join, workers)


def _run_variants(
    cfg: Config,
    variants: list[AdaptiveVariant],
    name: str,
    out_dir: Path | None,
    join: list[str] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Adaptive variants on the main backtest's seed sets and budget, in `<out_dir>/<name>/`.

    The main backtest's runs (adaptive at planner.beta and every baseline) and the runs of any
    sweep named in `join` are included for comparison when they exist.
    """
    bt = cfg.backtest
    out_dir = out_dir or bt.out_dir
    sweep_dir = out_dir / name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    ref_path = None
    for ss in bt.seed_sets:
        ss_cfg = _seed_set_cfg(cfg, ss, out_dir)
        prepare(ss_cfg)
        ref_path = ref_path or reference_blocks(cfg, ss_cfg.data.processed_dir, out_dir)
        for v in variants:
            planner = ss_cfg.planner.model_copy(update=v.model_dump())
            v_cfg = ss_cfg.model_copy(update={"planner": planner})
            run_dir = sweep_dir / "runs" / ss.name / v.label
            jobs.append(
                _Job(name, v_cfg, ss.name, "adaptive", cfg.seed, run_dir, v.label, ref_path)
            )
    frames = _execute(jobs, workers, cfg.log_level)
    results = pd.concat(frames, ignore_index=True)
    others = [out_dir / "results.parquet"] + [out_dir / j / "results.parquet" for j in join or []]
    for path in others:
        if not path.exists():
            continue
        other = pd.read_parquet(path)
        other.loc[other["strategy"] == "adaptive", "strategy"] = AdaptiveVariant(
            beta=cfg.planner.beta,
            gamma=cfg.planner.gamma,
            delta=cfg.planner.delta,
            ore_weight=cfg.planner.ore_weight,
        ).label
        other = other[~other["strategy"].isin(set(results["strategy"]))]
        results = pd.concat([other, results], ignore_index=True)
    results.to_parquet(sweep_dir / "results.parquet", index=False)
    summary = summarize_backtest(results, cfg, with_estimates=False)
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_backtest(results, cfg, sweep_dir / f"{name}.html")
    return results


def summarize_backtest(results: pd.DataFrame, cfg: Config, with_estimates: bool = True) -> dict:
    """Per strategy: metric at fixed metre checkpoints and metres to the F1 target."""
    budget = cfg.backtest.metre_budget
    checkpoints = np.array(cfg.backtest.report_at_m or [0.25 * budget, 0.5 * budget, budget])
    out: dict = {
        "cutoff_pct": cfg.model.cutoff_pct,
        "metre_budget": budget,
        "f1_target": cfg.backtest.f1_target,
        "block_f1_targets": cfg.backtest.block_f1_targets,
        "strategies": {},
    }
    for strategy, g in results.groupby("strategy"):
        runs = [r.sort_values("iteration") for _, r in g.groupby(["seed_set", "rng_seed"])]
        entry: dict = {"runs": len(runs)}
        for metric in CURVE_METRICS + BLOCK_METRICS:
            if metric not in g or g[metric].isna().all():
                continue
            if metric in BLOCK_METRICS:
                vals = np.array([_block_at(r, metric, checkpoints) for r in runs])
            else:
                vals = np.array([on_metre_grid(r, metric, checkpoints) for r in runs])
            entry[metric] = {
                f"at_{int(c)}m": {
                    "mean": float(np.nanmean(vals[:, i])),
                    "min": float(np.nanmin(vals[:, i])),
                    "max": float(np.nanmax(vals[:, i])),
                }
                for i, c in enumerate(checkpoints)
            }
        to_target = [metres_to_target(r, "f1_undrilled", cfg.backtest.f1_target) for r in runs]
        holes_to_target = [
            float(r.loc[r["f1_undrilled"] >= cfg.backtest.f1_target, "holes"].iloc[0])
            if (r["f1_undrilled"] >= cfg.backtest.f1_target).any()
            else float("nan")
            for r in runs
        ]
        entry["metres_to_f1_target"] = {
            "mean": float(np.nanmean(to_target)) if np.isfinite(to_target).any() else None,
            "reached": int(np.isfinite(to_target).sum()),
        }
        entry["holes_to_f1_target_mean"] = (
            float(np.nanmean(holes_to_target)) if np.isfinite(holes_to_target).any() else None
        )
        if "block_f1" in g and g["block_f1"].notna().any():
            entry["metres_to_block_f1"] = {}
            for target in cfg.backtest.block_f1_targets:
                m = np.array(
                    [
                        metres_to_target(r.dropna(subset=["block_f1"]), "block_f1", target)
                        for r in runs
                    ]
                )
                hit = np.isfinite(m).any()
                entry["metres_to_block_f1"][f"{target:g}"] = {
                    "mean": float(np.nanmean(m)) if hit else None,
                    "min": float(np.nanmin(m)) if hit else None,
                    "max": float(np.nanmax(m)) if hit else None,
                    "reached": int(np.isfinite(m).sum()),
                }
        out["strategies"][strategy] = entry
    if with_estimates:
        out["environment_estimates"] = _env_estimates(out, cfg)
        out["discovery_equivalent"] = _discovery_equivalent(results, cfg)
    return out


def _block_at(curve: pd.DataFrame, metric: str, checkpoints: np.ndarray) -> np.ndarray:
    """Block metric at the first evaluation at or after each checkpoint (NaN if none)."""
    scored = curve.dropna(subset=[metric])
    x, y = scored["cum_metres"].to_numpy(), scored[metric].to_numpy()
    idx = np.searchsorted(x, checkpoints, side="left")
    return np.where(idx < len(x), y[np.clip(idx, 0, len(y) - 1)], np.nan)


def _discovery_equivalent(results: pd.DataFrame, cfg: Config) -> dict:
    """Metres adaptive needed to match each baseline's ore metres at the full budget.

    Per seed set, the target is the baseline's mean ore metres at the budget (mean over RNG
    seeds for random); savings are averaged across seed sets. Environmental figures are
    ESTIMATES from the placeholder factors.
    """
    if "adaptive" not in set(results["strategy"]):
        return {}
    budget = cfg.backtest.discovery_at_m or cfg.backtest.metre_budget
    env = cfg.env
    out: dict = {"label": "ESTIMATE", "factors_note": env.source_note, "at_m": budget, "vs": {}}
    for baseline in sorted(set(results["strategy"]) - {"adaptive"}):
        per_set = []
        for seed_set, g in results.groupby("seed_set"):
            base = [
                r.sort_values("iteration")
                for _, r in g[g["strategy"] == baseline].groupby("rng_seed")
            ]
            adapt = g[g["strategy"] == "adaptive"].sort_values("iteration")
            if not base or adapt.empty:
                continue
            target = float(
                np.nanmean([on_metre_grid(r, "ore_m", np.array([budget]))[0] for r in base])
            )
            base_holes = float(
                np.mean([r.loc[r["cum_metres"] <= budget, "holes"].max() for r in base])
            )
            metres = metres_to_target(adapt, "ore_m", target)
            holes = (
                float(adapt.loc[adapt["cum_metres"] >= metres, "holes"].iloc[0])
                if np.isfinite(metres)
                else float("nan")
            )
            per_set.append((seed_set, target, metres, holes, base_holes))
        if not per_set:
            continue
        metres = np.array([m for _, _, m, _, _ in per_set])
        holes_saved = np.array([b - h for _, _, _, h, b in per_set])
        saved = budget - metres
        out["vs"][baseline] = {
            "baseline_ore_m": {s: t for s, t, _, _, _ in per_set},
            "adaptive_metres_to_match": {s: m for s, _, m, _, _ in per_set},
            "metres_saved_mean": float(np.nanmean(saved)),
            "metres_saved_min": float(np.nanmin(saved)),
            "metres_saved_max": float(np.nanmax(saved)),
            "holes_saved_mean": float(np.nanmean(holes_saved)),
            "water_l_estimate": float(np.nanmean(saved)) * env.water_l_per_m,
            "diesel_l_estimate": float(np.nanmean(saved)) * env.diesel_l_per_m,
            "pad_area_m2_estimate": float(np.nanmean(holes_saved)) * env.pad_area_m2_per_hole,
        }
    return out


def _env_estimates(summary: dict, cfg: Config) -> dict:
    """ESTIMATES of drilling avoided by adaptive vs each baseline, reaching the same F1 target."""
    s = summary["strategies"]
    if "adaptive" not in s or s["adaptive"]["metres_to_f1_target"]["mean"] is None:
        return {"note": "adaptive did not reach the F1 target within budget"}
    a_m = s["adaptive"]["metres_to_f1_target"]["mean"]
    a_h = s["adaptive"]["holes_to_f1_target_mean"]
    env = cfg.env
    out = {"label": "ESTIMATE", "factors_note": env.source_note, "vs": {}}
    for name, entry in s.items():
        b_m = entry["metres_to_f1_target"]["mean"]
        if name == "adaptive" or b_m is None:
            continue
        metres = b_m - a_m
        holes = (entry["holes_to_f1_target_mean"] or 0) - (a_h or 0)
        out["vs"][name] = {
            "metres_avoided": metres,
            "holes_avoided": holes,
            "water_l_estimate": metres * env.water_l_per_m,
            "diesel_l_estimate": metres * env.diesel_l_per_m,
            "pad_area_m2_estimate": holes * env.pad_area_m2_per_hole,
        }
    return out


COLORS = {
    "adaptive": "#2a6fdb",
    "grid": "#d9822b",
    "historical": "#7a7a7a",
    "random": "#3aa676",
    "adaptive-b0.5": "#2a6fdb",
    "adaptive-b1": "#6a51a3",
    "adaptive-b2": "#c2185b",
    "adaptive-b4": "#e6550d",
    "adaptive-b0.5-g1": "#08519c",
    "adaptive-b0.5-g2": "#000000",
    "adaptive-b1-g1": "#9e0142",
    "adaptive-b0.5-d1": "#1b9e77",
    "adaptive-b0.5-d2": "#66a61e",
    "adaptive-b1-d1": "#e7298a",
    "adaptive-o0-b0-d1": "#a6761d",
}


def plot_backtest(results: pd.DataFrame, cfg: Config, path: Path) -> None:
    grid = np.linspace(0, cfg.backtest.metre_budget, 101)
    has_blocks = "block_f1" in results and results["block_f1"].notna().any()
    metrics = ["ore_m", "f1_undrilled_eg"] + (["block_f1"] if has_blocks else [])
    titles = [
        f"Discovery: ore metres intersected (Cu > {cfg.model.cutoff_pct}%)",
        "Delineation: F1 on undrilled composites (estimated grade > cutoff)",
        "Delineation: block F1 vs the all-holes reference model",
    ]
    if has_blocks:
        runs_of = results.groupby(["strategy", "seed_set", "rng_seed"])["block_f1"]
        results = results.assign(block_f1=runs_of.ffill())
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=titles[: len(metrics)])
    for col, metric in enumerate(metrics, start=1):
        for strategy, g in results.groupby("strategy"):
            runs = [r.sort_values("iteration") for _, r in g.groupby(["seed_set", "rng_seed"])]
            vals = np.array([on_metre_grid(r, metric, grid) for r in runs])
            mean, lo, hi = (np.nanmean(vals, 0), np.nanmin(vals, 0), np.nanmax(vals, 0))
            color = COLORS.get(strategy, "#444")
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([grid, grid[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=mean,
                    name=f"{strategy} (n={len(runs)})",
                    line={"color": color, "width": 2.5},
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Cumulative metres drilled", row=1, col=col)
    fig.update_layout(
        title="Hillside backtest: mean and range across seed sets (and RNG seeds for random)",
        template="plotly_white",
        height=480,
    )
    fig.write_html(path, include_plotlyjs="cdn")
