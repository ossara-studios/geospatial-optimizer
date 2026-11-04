"""Plotly figures for the visualizer. Pure functions of planner-visible tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DRILLED = "#2a6fdb"
SEED = "#7a7a7a"
CANDIDATE = "#c9c9c9"
LATEST = "#d62728"


def plan_map(
    candidates: pd.DataFrame,
    seed_collars: pd.DataFrame,
    records: pd.DataFrame,
    limits: dict,
    grid: pd.DataFrame | None = None,
    z_level: float | None = None,
) -> go.Figure:
    """Plan view: P(ore) slice (optional), no-go zones, candidates, seed and drilled collars."""
    fig = go.Figure()
    if grid is not None and len(grid):
        zs = np.sort(grid["z"].unique())
        z = zs[np.argmin(np.abs(zs - (z_level if z_level is not None else zs[len(zs) // 2])))]
        sl = grid[grid["z"] == z]
        piv = sl.pivot_table(index="y", columns="x", values="p_ore")
        fig.add_trace(
            go.Heatmap(
                x=piv.columns,
                y=piv.index,
                z=piv.to_numpy(),
                zmin=0,
                zmax=1,
                colorscale="YlOrRd",
                colorbar={"title": "P(ore)"},
                hovertemplate="x=%{x:.0f} y=%{y:.0f} P(ore)=%{z:.2f}<extra></extra>",
                name=f"P(ore) at z={z:.0f} m",
            )
        )
    for zone in limits.get("no_go_zones", []):
        if zone["kind"] == "rect":
            fig.add_shape(
                type="rect",
                x0=zone["x_min"],
                x1=zone["x_max"],
                y0=zone["y_min"],
                y1=zone["y_max"],
                line={"color": "black", "dash": "dot"},
                fillcolor="rgba(0,0,0,0.08)",
            )
        else:
            fig.add_shape(
                type="circle",
                x0=zone["x"] - zone["radius"],
                x1=zone["x"] + zone["radius"],
                y0=zone["y"] - zone["radius"],
                y1=zone["y"] + zone["radius"],
                line={"color": "black", "dash": "dot"},
                fillcolor="rgba(0,0,0,0.08)",
            )
    b = limits.get("site_bounds")
    if b:
        fig.add_shape(
            type="rect",
            x0=b["x_min"],
            x1=b["x_max"],
            y0=b["y_min"],
            y1=b["y_max"],
            line={"color": "black", "width": 1},
        )
    fig.add_trace(_collars(candidates, "Candidates", CANDIDATE, 5))
    fig.add_trace(_collars(seed_collars, "Seed holes", SEED, 6))
    drilled = _drilled(records, candidates)
    if len(drilled):
        fig.add_trace(_collars(drilled, "Drilled by planner", DRILLED, 9, text="iteration"))
        fig.add_trace(_collars(drilled.tail(1), "Latest", LATEST, 15, symbol="star"))
    rejected = _rejected(records, candidates)
    if len(rejected):
        fig.add_trace(_collars(rejected, "Rejected / denied", "black", 8, symbol="x"))
    fig.update_layout(
        template="plotly_white",
        height=620,
        xaxis={"title": "East (m, local)", "scaleanchor": "y"},
        yaxis={"title": "North (m, local)"},
        legend={"orientation": "h", "y": -0.12},
        margin={"t": 30},
    )
    return fig


def section(
    grid: pd.DataFrame, composites: pd.DataFrame, x_level: float, cutoff: float
) -> go.Figure:
    """North-south section of P(ore) at an easting, with nearby assayed composites."""
    xs = np.sort(grid["x"].unique())
    x = xs[np.argmin(np.abs(xs - x_level))]
    sl = grid[grid["x"] == x]
    piv = sl.pivot_table(index="z", columns="y", values="p_ore")
    fig = go.Figure(
        go.Heatmap(
            x=piv.columns,
            y=piv.index,
            z=piv.to_numpy(),
            zmin=0,
            zmax=1,
            colorscale="YlOrRd",
            colorbar={"title": "P(ore)"},
        )
    )
    if len(composites):
        step = xs[1] - xs[0] if len(xs) > 1 else 50.0
        near = composites[np.abs(composites["x"] - x) <= step / 2]
        fig.add_trace(
            go.Scatter(
                x=near["y"],
                y=near["z"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": np.where(near["cu"] > cutoff, "black", "white"),
                    "line": {"color": "black", "width": 0.5},
                },
                name=f"Composites within {step / 2:.0f} m (black: Cu > {cutoff}%)",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        title=f"Section at x = {x:.0f} m",
        xaxis_title="North (m)",
        yaxis_title="Elevation (m)",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def progress(records: pd.DataFrame) -> go.Figure:
    drilled = records[records["metres_drilled"] > 0]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=drilled["iteration"], y=drilled["exp_ore_m"], name="Expected ore m (model)")
    )
    fig.add_trace(
        go.Scatter(
            x=drilled["iteration"],
            y=drilled["cumulative_metres"],
            name="Cumulative metres",
            yaxis="y2",
            line={"color": DRILLED},
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        xaxis_title="Iteration",
        yaxis={"title": "Expected ore m"},
        yaxis2={"title": "Cumulative m", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.3},
        margin={"t": 20},
    )
    return fig


def _drilled(records: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["x", "y", "hole_id", "iteration"])
    r = records[records["metres_drilled"] > 0][["hole_idx", "iteration"]]
    return r.merge(candidates[["hole_idx", "hole_id", "x", "y"]], on="hole_idx")


def _rejected(records: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["x", "y", "hole_id"])
    bad = (records["check_status"] == "REJECTED") | (records["decision"] == "DENIED")
    r = records[bad][["hole_idx", "iteration"]]
    return r.merge(candidates[["hole_idx", "hole_id", "x", "y"]], on="hole_idx")


def _collars(
    df: pd.DataFrame, name: str, color: str, size: int, symbol: str = "circle", text: str = ""
) -> go.Scatter:
    hover = df["hole_id"].astype(str) if "hole_id" in df else None
    if text and text in df:
        hover = df["hole_id"].astype(str) + " (iter " + df[text].astype(str) + ")"
    return go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        name=name,
        text=hover,
        hoverinfo="text",
        marker={"color": color, "size": size, "symbol": symbol, "line": {"width": 0.5}},
    )
