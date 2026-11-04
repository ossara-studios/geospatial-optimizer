"""Streamlit visualizer: `geoopt viz`. Reads run artifacts only; works while a live run writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from geoopt.config import DEFAULT_CONFIG_PATH, load_config
from geoopt.viz import data, figures


def _config_path() -> Path:
    args = sys.argv[1:]
    return Path(args[args.index("--config") + 1]) if "--config" in args else DEFAULT_CONFIG_PATH


def run_view(run_dir: Path) -> None:
    run = data.open_run(run_dir)
    recs = data.records(run)
    tables = data.planner_tables(run)
    cfg = run.config
    cutoff = cfg["model"]["cutoff_pct"]

    st.caption(
        f"strategy **{cfg.get('strategy')}** · seed holes {len(tables['seed_collars'])} · "
        f"candidates {len(tables['candidates'])} · cutoff Cu > {cutoff}% (placeholder)"
    )
    if recs.empty:
        st.info("No iterations written yet.")
        return

    drilled = recs[recs["metres_drilled"] > 0]
    c = st.columns(4)
    c[0].metric("Iterations", len(recs))
    c[1].metric("Holes drilled", len(drilled))
    c[2].metric("Metres drilled", f"{drilled['metres_drilled'].sum():,.0f}")
    c[3].metric(
        "Rejected / denied",
        int(((recs["check_status"] == "REJECTED") | (recs["decision"] == "DENIED")).sum()),
    )

    iteration = st.select_slider("Iteration", options=run.iterations, value=run.iterations[-1])
    upto = recs[recs["iteration"] <= iteration]
    grid = data.grid(run, iteration)
    z_level = None
    if grid is not None:
        zs = sorted(grid["z"].unique())
        z_level = st.select_slider(
            "Plan slice elevation (m)",
            options=zs,
            value=zs[len(zs) // 2],
            format_func=lambda v: f"{v:.0f}",
        )
    st.plotly_chart(
        figures.plan_map(
            tables["candidates"], tables["seed_collars"], upto, cfg["limits"], grid, z_level
        ),
        width="stretch",
    )

    latest = upto.iloc[-1]
    left, right = st.columns([2, 3])
    with left:
        st.subheader(f"Iteration {int(latest['iteration'])}: {latest['hole_id']}")
        st.write(
            f"ReqId {int(latest['req_id'])} · check **{latest['check_status']}** "
            f"({latest['check_reason']}) · decision **{latest['decision']}**"
        )
        st.write(
            f"Mean P(ore) {latest['prob_ore']:.2f} · expected ore {latest['exp_ore_m']:.1f} m · "
            f"depth {latest['checked_depth']:.0f} m · drilled {latest['metres_drilled']:.0f} m"
        )
        bd = latest.get("breakdown")
        if isinstance(bd, dict) and bd:
            st.caption("Score breakdown")
            st.dataframe(pd.Series(bd, name="value").round(3), width="stretch")
    with right:
        if len(drilled):
            st.plotly_chart(figures.progress(upto), width="stretch")

    if grid is not None:
        xs = sorted(grid["x"].unique())
        x_level = st.select_slider(
            "Section easting (m)",
            options=xs,
            value=xs[len(xs) // 2],
            format_func=lambda v: f"{v:.0f}",
        )
        lab = data.drilled_composites(
            run, upto.loc[upto["metres_drilled"] > 0, "hole_idx"].astype(int).tolist()
        )
        comps = pd.concat([tables["seed_composites"], lab], ignore_index=True)
        st.plotly_chart(figures.section(grid, comps, x_level, cutoff), width="stretch")

    with st.expander("Iteration log"):
        cols = [
            "iteration",
            "req_id",
            "hole_id",
            "check_status",
            "check_reason",
            "decision",
            "checked_depth",
            "metres_drilled",
            "cumulative_metres",
            "score",
            "exp_ore_m",
        ]
        st.dataframe(recs[cols], width="stretch", hide_index=True)


def backtest_view(out_dir: Path) -> None:
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "backtest.html"
    if not summary_path.exists():
        st.info(f"No backtest results in {out_dir}. Run `geoopt backtest`.")
        return
    summary = json.loads(summary_path.read_text())
    if html_path.exists():
        st.components.v1.html(html_path.read_text(), height=520, scrolling=False)
    rows = []
    for name, s in summary["strategies"].items():
        rows.append(
            {
                "strategy": name,
                "runs": s["runs"],
                **{f"ore_m {k}": v["mean"] for k, v in s["ore_m"].items()},
                **{f"F1 {k}": v["mean"] for k, v in s["f1_undrilled"].items()},
                "metres to F1 target": s["metres_to_f1_target"]["mean"],
            }
        )
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    env = summary.get("environment_estimates", {})
    if env.get("vs"):
        st.caption(f"ESTIMATES. {env.get('factors_note', '')}")
        st.dataframe(pd.DataFrame(env["vs"]).T.round(0), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Geospatial Optimizer", layout="wide")
    cfg = load_config(_config_path())
    st.title("Geospatial Optimizer")
    tab_run, tab_bt = st.tabs(["Run", "Backtest"])
    with tab_run:
        runs = data.list_runs(cfg.loop.runs_dir)
        if not runs:
            st.info(f"No runs in {cfg.loop.runs_dir}. Start one with `geoopt run-loop`.")
        else:
            choice = st.selectbox("Run", runs, format_func=lambda p: p.name)
            live = st.toggle("Live refresh (2 s)", value=False)

            @st.fragment(run_every=2.0 if live else None)
            def _body() -> None:
                run_view(choice)

            _body()
    with tab_bt:
        backtest_view(cfg.backtest.out_dir)


main()
