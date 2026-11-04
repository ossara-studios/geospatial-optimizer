"""Command-line entry points."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from geoopt.backtest import run_backtest, run_beta_sweep
from geoopt.config import DEFAULT_CONFIG_PATH, load_config
from geoopt.data.load import DataConfigIncomplete
from geoopt.data.prepare import prepare as prepare_data
from geoopt.emulator.server import auto_approve, cli_approve, serve
from geoopt.logging_setup import configure_logging
from geoopt.run import run_inprocess, run_opcua, run_rig_sim, summarize

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
    bridge: Annotated[str, typer.Option(help="inprocess | opcua")] = "inprocess",
    max_holes: Annotated[int | None, typer.Option(help="Override loop.max_holes.")] = None,
) -> None:
    """Run the decision loop end to end."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if max_holes is not None:
        cfg = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"max_holes": max_holes})})
    if bridge == "inprocess":
        records = asyncio.run(run_inprocess(cfg, strategy))
    elif bridge == "opcua":
        records = asyncio.run(run_opcua(cfg, strategy))
    else:
        raise typer.BadParameter(f"unknown bridge {bridge!r}")
    typer.echo(summarize(records))


@app.command()
def backtest(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    beta_sweep: Annotated[
        bool, typer.Option("--beta-sweep", help="Adaptive only, at each backtest.beta_sweep.")
    ] = False,
) -> None:
    """Run every strategy over every seed set with the same metre budget."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    if beta_sweep:
        run_beta_sweep(cfg)
        typer.echo(f"wrote {cfg.backtest.out_dir / 'beta_sweep'}")
        return
    run_backtest(cfg)
    typer.echo(f"wrote {cfg.backtest.out_dir}")


@app.command("rig-sim")
def rig_sim(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the rig + lab simulator as an OPC UA client of EAE (or the emulator)."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_rig_sim(cfg))


@app.command()
def emulator(
    config: Annotated[Path, typer.Option(help="Main config file.")] = DEFAULT_CONFIG_PATH,
    auto_approve_all: Annotated[
        bool,
        typer.Option("--auto-approve", help="Approve every checked request without prompting."),
    ] = False,
) -> None:
    """Run the EAE emulator (OPC UA server with the EAE node map and limit logic)."""
    cfg = load_config(config)
    configure_logging(cfg.log_level)
    approver = auto_approve if auto_approve_all else cli_approve
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(cfg, approver))


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
