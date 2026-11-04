"""Retrospective backtest: every strategy, same seed set, same candidates, same metre budget.

Evaluation sits outside the planner: it reads held-out truth through LabSim to score each
run, but never feeds it to the model or the strategy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geoopt.config import Config, SeedSet
from geoopt.data.prepare import prepare
from geoopt.loop import IterationRecord
from geoopt.metrics import classification, metres_to_target, on_metre_grid, ore_metres
from geoopt.run import build_inprocess_loop

log = logging.getLogger(__name__)

CURVE_METRICS = ["ore_m", "f1_undrilled", "bacc_undrilled", "f1_all"]


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


async def _one_run(
    cfg: Config, seed_set: str, strategy: str, rng_seed: int, run_dir: Path, label: str = ""
) -> pd.DataFrame:
    loop, lab = build_inprocess_loop(cfg, strategy, run_dir, rng_seed=rng_seed, write_grid=False)
    truth = lab.truth_for_scoring()
    xyz = truth[["x", "y", "z"]].to_numpy()
    actual = (truth["cu"] > cfg.model.cutoff_pct).to_numpy()
    cutoff, threshold = cfg.model.cutoff_pct, cfg.backtest.ore_threshold
    rows: list[dict] = []
    found = {"ore_m": 0.0}

    def score(record: IterationRecord | None) -> None:
        if record is not None and record.metres_drilled > 0:
            released = pd.read_parquet(run_dir / "lab" / f"{record.hole_idx}.parquet")
            found["ore_m"] += ore_metres(released, cutoff)
        undrilled = ~truth["hole_idx"].isin(loop.drilled_holes).to_numpy()
        predicted = loop.model.prob_ore(xyz) > threshold
        und = classification(predicted[undrilled], actual[undrilled])
        rows.append(
            {
                "seed_set": seed_set,
                "strategy": label or strategy,
                "beta": cfg.planner.beta if strategy == "adaptive" else float("nan"),
                "rng_seed": rng_seed,
                "iteration": 0 if record is None else record.iteration,
                "holes": len(loop.drilled_holes),
                "cum_metres": 0.0 if record is None else record.cumulative_metres,
                "ore_m": found["ore_m"],
                "f1_undrilled": und["f1"],
                "bacc_undrilled": und["balanced_accuracy"],
                "f1_all": classification(predicted, actual)["f1"],
                "n_undrilled": und["n"],
            }
        )

    score(None)
    loop.on_iteration = score
    await loop.run()
    return pd.DataFrame(rows)


def run_backtest(cfg: Config, out_dir: Path | None = None) -> pd.DataFrame:
    bt = cfg.backtest
    out_dir = out_dir or bt.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for ss in bt.seed_sets:
        ss_cfg = _seed_set_cfg(cfg, ss, out_dir)
        prepare(ss_cfg)
        for strategy in bt.strategies:
            seeds = bt.random_seeds if strategy == "random" else [cfg.seed]
            for rng_seed in seeds:
                t0 = time.perf_counter()
                run_dir = out_dir / "runs" / ss.name / f"{strategy}-{rng_seed}"
                df = asyncio.run(_one_run(ss_cfg, ss.name, strategy, rng_seed, run_dir))
                frames.append(df)
                last = df.iloc[-1]
                log.info(
                    "backtest seed_set=%s strategy=%s rng=%s holes=%d metres=%.0f ore_m=%.0f "
                    "f1_undrilled=%.3f seconds=%.0f",
                    ss.name,
                    strategy,
                    rng_seed,
                    last["holes"],
                    last["cum_metres"],
                    last["ore_m"],
                    last["f1_undrilled"],
                    time.perf_counter() - t0,
                )
    results = pd.concat(frames, ignore_index=True)
    results.to_parquet(out_dir / "results.parquet", index=False)
    summary = summarize_backtest(results, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_backtest(results, cfg, out_dir / "backtest.html")
    return results


def run_beta_sweep(cfg: Config, out_dir: Path | None = None) -> pd.DataFrame:
    """Adaptive at each `backtest.beta_sweep` beta, on the main backtest's seed sets and budget.

    Writes `<out_dir>/beta_sweep/`; if the main backtest's results.parquet exists, its runs
    (adaptive at planner.beta and every baseline) are included for comparison.
    """
    bt = cfg.backtest
    out_dir = out_dir or bt.out_dir
    sweep_dir = out_dir / "beta_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for ss in bt.seed_sets:
        ss_cfg = _seed_set_cfg(cfg, ss, out_dir)
        prepare(ss_cfg)
        for beta in bt.beta_sweep:
            label = f"adaptive-b{beta:g}"
            b_cfg = ss_cfg.model_copy(
                update={"planner": ss_cfg.planner.model_copy(update={"beta": beta})}
            )
            t0 = time.perf_counter()
            run_dir = sweep_dir / "runs" / ss.name / label
            df = asyncio.run(_one_run(b_cfg, ss.name, "adaptive", cfg.seed, run_dir, label))
            frames.append(df)
            last = df.iloc[-1]
            log.info(
                "beta_sweep seed_set=%s beta=%g holes=%d metres=%.0f ore_m=%.0f "
                "f1_undrilled=%.3f seconds=%.0f",
                ss.name,
                beta,
                last["holes"],
                last["cum_metres"],
                last["ore_m"],
                last["f1_undrilled"],
                time.perf_counter() - t0,
            )
    results = pd.concat(frames, ignore_index=True)
    main = out_dir / "results.parquet"
    if main.exists():
        base = pd.read_parquet(main)
        base.loc[base["strategy"] == "adaptive", "strategy"] = f"adaptive-b{cfg.planner.beta:g}"
        results = pd.concat([base, results], ignore_index=True)
    results.to_parquet(sweep_dir / "results.parquet", index=False)
    summary = summarize_backtest(results, cfg, with_estimates=False)
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_backtest(results, cfg, sweep_dir / "beta_sweep.html")
    return results


def summarize_backtest(results: pd.DataFrame, cfg: Config, with_estimates: bool = True) -> dict:
    """Per strategy: metric at fixed metre checkpoints and metres to the F1 target."""
    budget = cfg.backtest.metre_budget
    checkpoints = np.array([0.25, 0.5, 1.0]) * budget
    out: dict = {
        "cutoff_pct": cfg.model.cutoff_pct,
        "metre_budget": budget,
        "f1_target": cfg.backtest.f1_target,
        "strategies": {},
    }
    for strategy, g in results.groupby("strategy"):
        runs = [r.sort_values("iteration") for _, r in g.groupby(["seed_set", "rng_seed"])]
        entry: dict = {"runs": len(runs)}
        for metric in ("ore_m", "f1_undrilled", "bacc_undrilled"):
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
        out["strategies"][strategy] = entry
    if with_estimates:
        out["environment_estimates"] = _env_estimates(out, cfg)
        out["discovery_equivalent"] = _discovery_equivalent(results, cfg)
    return out


def _discovery_equivalent(results: pd.DataFrame, cfg: Config) -> dict:
    """Metres adaptive needed to match each baseline's ore metres at the full budget.

    Per seed set, the target is the baseline's mean ore metres at the budget (mean over RNG
    seeds for random); savings are averaged across seed sets. Environmental figures are
    ESTIMATES from the placeholder factors.
    """
    if "adaptive" not in set(results["strategy"]):
        return {}
    budget = cfg.backtest.metre_budget
    env = cfg.env
    out: dict = {"label": "ESTIMATE", "factors_note": env.source_note, "vs": {}}
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
}


def plot_backtest(results: pd.DataFrame, cfg: Config, path: Path) -> None:
    grid = np.linspace(0, cfg.backtest.metre_budget, 101)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Discovery: ore metres intersected (Cu > {cfg.model.cutoff_pct}%)",
            "Delineation: F1 on undrilled held-out composites",
        ),
    )
    for col, metric in enumerate(("ore_m", "f1_undrilled"), start=1):
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
