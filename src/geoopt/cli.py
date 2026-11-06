"""Command-line entry points."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import subprocess
import sys
import webbrowser
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
from geoopt.data.prepare import prepared_paths
from geoopt.logging_setup import configure_logging
from geoopt.progress import Progress
from geoopt.run import run_inprocess, summarize

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


def _check_port(port: int) -> None:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            typer.echo(f"error: port {port} is already in use; pick another with --port", err=True)
            raise typer.Exit(1)


def _ensure_demo_inputs(cfg: Config) -> None:
    """Prepare the data and build the rule-of-thumb baseline if this machine has neither yet."""
    if not prepared_paths(cfg.data.processed_dir).meta.exists():
        typer.echo("first run: preparing the drillhole data ...")
        try:
            prepare_data(cfg)
        except (DataConfigIncomplete, FileNotFoundError) as exc:
            typer.echo(
                f"error: {exc}\nThe raw drillhole data isn't on this machine: copy "
                "data/raw/hillside/ from one that has it, or rebuild it (scripts/fetch_data.md).",
                err=True,
            )
            raise typer.Exit(1) from exc
    baseline_file = cfg.kpi.baseline_file
    if baseline_file is not None and not baseline_file.exists():
        typer.echo("first run: building the rule-of-thumb baseline ...")
        near_ore_curve(cfg, baseline_file)


@app.command()
def demo(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    holes: Annotated[
        int | None, typer.Option(help="Holes to drill (default loop.max_holes).")
    ] = None,
    speed: Annotated[
        float | None, typer.Option(help="Drilling speed, m/s (default rig.metres_per_second).")
    ] = None,
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8600,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the page.")] = True,
) -> None:
    """Run the planner at demo pace and watch it live in the browser."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if holes is not None:
        cfg = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"max_holes": holes})})
    if speed is not None:
        rig = cfg.rig.model_copy(update={"metres_per_second": speed})
        cfg = cfg.model_copy(update={"rig": rig})
    _check_port(port)
    _ensure_demo_inputs(cfg)
    progress = Progress()
    url = f"http://localhost:{port}"

    async def job() -> None:
        if open_browser:
            await asyncio.sleep(1.0)  # let the server start listening first
            webbrowser.open(url)
        records = await run_inprocess(cfg, "adaptive", progress=progress)
        s = summarize(records)
        typer.echo(
            f"demo done: {s['holes_drilled']} holes, {s['metres']:,.0f} m. "
            "The page stays up; Ctrl+C to stop."
        )

    typer.echo(f"live view on {url}  (Ctrl+C to stop)")
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
    _check_port(port)
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
