"""Command-line entry points."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer

from geoopt.backtest import (
    run_assay_delay_sweep,
    run_backtest,
    run_beta_sweep,
    run_boundary_sweep,
    run_delineation_sweep,
)
from geoopt.baseline import near_ore_curve
from geoopt.config import DEFAULT_CONFIG_PATH, Config, load_config
from geoopt.data.load import DataConfigIncomplete
from geoopt.data.prepare import prepare as prepare_data
from geoopt.logging_setup import configure_logging
from geoopt.progress import Progress
from geoopt.run import new_run_dir, run_inprocess, summarize

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Geospatial Optimizer."""


@app.command()
def prepare(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
) -> None:
    """Load raw drillhole data and write the processed seed/candidate/truth tables."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    try:
        paths = prepare_data(cfg)
    except DataConfigIncomplete as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {paths.meta.parent}")


@app.command("run-loop")
def run_loop(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    strategy: Annotated[
        str, typer.Option(help="adaptive | random | grid | historical")
    ] = "adaptive",
    max_holes: Annotated[int | None, typer.Option(help="Override loop.max_holes.")] = None,
) -> None:
    """Run the decision loop end to end, as fast as it goes (no live view)."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if max_holes is not None:
        cfg = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"max_holes": max_holes})})
    records = asyncio.run(run_inprocess(cfg, strategy))
    typer.echo(summarize(records))


@app.command()
def backtest(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    beta_sweep: Annotated[
        bool, typer.Option("--beta-sweep", help="Adaptive only, at each backtest.beta_sweep.")
    ] = False,
    boundary_sweep: Annotated[
        bool,
        typer.Option("--boundary-sweep", help="Adaptive only, at each backtest.boundary_sweep."),
    ] = False,
    delineation_sweep: Annotated[
        bool,
        typer.Option(
            "--delineation-sweep", help="Adaptive only, at each backtest.delineation_sweep."
        ),
    ] = False,
    assay_delay_sweep: Annotated[
        bool,
        typer.Option(
            "--assay-delay-sweep",
            help="Adaptive at backtest.assay_delay_beta with each backtest.assay_delay_sweep lag.",
        ),
    ] = False,
    workers: Annotated[int, typer.Option(help="Parallel runs (processes).")] = 1,
) -> None:
    """Run every strategy over every seed set with the same metre budget."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if assay_delay_sweep:
        run_assay_delay_sweep(cfg, workers=workers)
        typer.echo(f"wrote {cfg.backtest.out_dir / 'assay_delay'}")
        return
    if delineation_sweep:
        run_delineation_sweep(cfg, workers=workers)
        typer.echo(f"wrote {cfg.backtest.out_dir / 'delineation_sweep'}")
        return
    if boundary_sweep:
        run_boundary_sweep(cfg, workers=workers)
        typer.echo(f"wrote {cfg.backtest.out_dir / 'boundary_sweep'}")
        return
    if beta_sweep:
        run_beta_sweep(cfg, workers=workers)
        typer.echo(f"wrote {cfg.backtest.out_dir / 'beta_sweep'}")
        return
    run_backtest(cfg, workers=workers)
    typer.echo(f"wrote {cfg.backtest.out_dir}")


@app.command()
def baseline(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
) -> None:
    """Replay the near-ore rule on the prepared data; run-loop reports cost saved against it."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if cfg.kpi.baseline_file is None:
        raise typer.BadParameter("kpi.baseline_file is not set")
    curve = near_ore_curve(cfg, cfg.kpi.baseline_file)
    last = curve.iloc[-1]
    typer.echo(
        f"near-ore baseline: {int(last.holes)} holes, {last.metres:,.0f} m, "
        f"{int(last.new_ore_blocks)} new ore blocks -> {cfg.kpi.baseline_file}"
    )


def _serve(
    cfg: Config,
    port: int,
    job: Callable[[], Awaitable[None]] | None = None,
    progress: Progress | None = None,
) -> None:
    import uvicorn

    from geoopt.live.server import build_app

    uvicorn.run(
        build_app(cfg, progress=progress, job=job), host="127.0.0.1", port=port, log_level="warning"
    )


@app.command()
def demo(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    max_holes: Annotated[int | None, typer.Option(help="Override loop.max_holes.")] = None,
    speed: Annotated[
        float | None, typer.Option(help="Drilling speed in m/s (overrides rig.metres_per_second).")
    ] = None,
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8600,
) -> None:
    """Run the planner at demo pace and show it live in the browser. One command, one process."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if max_holes is not None:
        cfg = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"max_holes": max_holes})})
    if speed is not None:
        rig = cfg.rig.model_copy(update={"metres_per_second": speed})
        cfg = cfg.model_copy(update={"rig": rig})
    progress = Progress()
    run_dir = new_run_dir(cfg, "adaptive")

    async def job() -> None:
        records = await run_inprocess(cfg, "adaptive", run_dir, progress=progress)
        typer.echo(summarize(records))

    typer.echo(f"live view on http://localhost:{port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        _serve(cfg, port, job, progress)


@app.command()
def live(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8600,
) -> None:
    """Live view of finished runs only: open /?replay to replay the latest one."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    typer.echo(f"live view on http://localhost:{port}/?replay")
    with contextlib.suppress(KeyboardInterrupt):
        _serve(cfg, port)


@app.command()
def viz(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    port: Annotated[int, typer.Option(help="Streamlit port.")] = 8501,
) -> None:
    """Open the run/backtest visualizer (Streamlit) in the browser."""
    app_path = Path(__file__).parent / "viz" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run([*cmd, "--", "--config", str(config.resolve())], check=False)
