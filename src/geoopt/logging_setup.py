"""Process-wide logging configuration."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
