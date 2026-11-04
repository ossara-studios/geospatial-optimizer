"""Phase 0 spike: one recommendation -> limit check -> operator approval over OPC UA.

Runs unchanged against the emulator or real EAE; point it with --endpoint (or config).

    python scripts/opcua_roundtrip.py --depth 650
    python scripts/opcua_roundtrip.py --endpoint opc.tcp://<eae-host>:4840 --depth 300
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from geoopt.bridge.opcua import roundtrip
from geoopt.config import DEFAULT_CONFIG_PATH, load_config
from geoopt.logging_setup import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--endpoint", help="Override opcua.endpoint from config.")
    parser.add_argument("--depth", type=float, default=300.0, help="Planned depth, m.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.endpoint:
        cfg = cfg.model_copy(
            update={"opcua": cfg.opcua.model_copy(update={"endpoint": args.endpoint})}
        )
    configure_logging(cfg.log_level)

    try:
        result = asyncio.run(roundtrip(cfg, args.depth))
    except TimeoutError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    decision = result.decision.name if result.decision else "n/a (rejected)"
    print(
        f"OK req_id={result.req_id} status={result.check.status.name} "
        f"reason={result.check.reason.name} depth={result.check.depth:.1f} decision={decision}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
