"""Command-line entry points."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Annotated

import typer

from geoopt.config import DEFAULT_CONFIG_PATH, load_config
from geoopt.emulator.server import auto_approve, cli_approve, serve
from geoopt.logging_setup import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Geospatial Optimizer."""


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
