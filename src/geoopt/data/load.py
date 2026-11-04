"""Raw drillhole tables -> validated canonical tables (see docs/data_schema.md).

The raw layout is never assumed: column names, units and conventions all come from
the `data` section of the config, filled in after inspecting the real files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoopt.config import DataConfig, TableSource

log = logging.getLogger(__name__)

FT_TO_M = 0.3048
PPM_PER_PCT = 10_000.0

REQUIRED = {
    "collars": ["hole_id", "x", "y", "z"],
    "surveys": ["hole_id", "depth", "azimuth", "dip"],
    "assays": ["hole_id", "from_m", "to_m", "cu"],
}
OPTIONAL = {"collars": ["total_depth", "azimuth", "dip", "drill_date"], "surveys": [], "assays": []}


class DataConfigIncomplete(RuntimeError):
    pass


@dataclass(frozen=True)
class DrillholeTables:
    collars: pd.DataFrame  # hole_id, x, y, z (local m), total_depth, azimuth, dip, [drill_date]
    surveys: pd.DataFrame  # hole_id, depth, azimuth, dip
    assays: pd.DataFrame  # hole_id, from_m, to_m, cu (% Cu, NaN = not analysed)
    origin: tuple[float, float, float]


def _require_complete(cfg: DataConfig) -> None:
    missing = [
        name
        for name in ("collars", "surveys", "assays", "depth_unit", "cu_unit", "dip_convention")
        if getattr(cfg, name) is None
    ]
    if missing:
        raise DataConfigIncomplete(
            f"data config not confirmed yet: {', '.join(missing)} are null. Inspect the raw "
            "files, document them in docs/data_schema.md, then fill in config/default.yaml."
        )


def _read_table(raw_dir: Path, src: TableSource, table: str) -> pd.DataFrame:
    path = raw_dir / src.file
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=src.sheet or 0)
    else:
        df = pd.read_csv(path)
    unknown = set(src.columns) - set(REQUIRED[table]) - set(OPTIONAL[table])
    if unknown:
        raise ValueError(f"{table}: unknown canonical columns {sorted(unknown)}")
    absent = [c for c in REQUIRED[table] if c not in src.columns]
    if absent:
        raise ValueError(f"{table}: no mapping for required columns {absent}")
    raw_missing = [raw for raw in src.columns.values() if raw not in df.columns]
    if raw_missing:
        raise ValueError(f"{table}: columns {raw_missing} not found in {path.name}")
    out = df[list(src.columns.values())].rename(columns={v: k for k, v in src.columns.items()})
    out["hole_id"] = out["hole_id"].astype(str).str.strip()
    return out


def clean_cu(values: pd.Series, unit: str, below_detection_factor: float) -> pd.Series:
    """Blank = not analysed (NaN, kept). Negative = below detection, stored as -DL."""
    cu = pd.to_numeric(values, errors="coerce").astype(float)
    bdl = cu < 0
    cu = cu.where(~bdl, -cu * below_detection_factor)
    if unit == "ppm":
        cu = cu / PPM_PER_PCT
    return cu


def load_tables(cfg: DataConfig) -> DrillholeTables:
    _require_complete(cfg)
    assert cfg.collars and cfg.surveys and cfg.assays  # narrowed by _require_complete
    collars = _read_table(cfg.raw_dir, cfg.collars, "collars")
    surveys = _read_table(cfg.raw_dir, cfg.surveys, "surveys")
    assays = _read_table(cfg.raw_dir, cfg.assays, "assays")
    to_m = FT_TO_M if cfg.depth_unit == "ft" else 1.0
    dip_sign = -1.0 if cfg.dip_convention == "positive_down" else 1.0

    # Collars.
    dupes = collars["hole_id"].duplicated()
    if dupes.any():
        raise ValueError(
            f"duplicate collar hole_ids: {collars.loc[dupes, 'hole_id'].tolist()[:10]}"
        )
    origin = cfg.origin or tuple(
        float(np.floor(collars[c].min() / 100.0) * 100.0) for c in ("x", "y", "z")
    )
    for axis, o in zip(("x", "y", "z"), origin, strict=True):
        collars[axis] = collars[axis].astype(float) - o
    if "drill_date" in collars:
        collars["drill_date"] = pd.to_datetime(collars["drill_date"], errors="coerce")

    # Surveys.
    surveys["depth"] = surveys["depth"].astype(float) * to_m
    surveys["dip"] = surveys["dip"].astype(float) * dip_sign
    surveys = surveys.dropna(subset=["depth", "azimuth", "dip"])

    # Assays.
    assays["from_m"] = assays["from_m"].astype(float) * to_m
    assays["to_m"] = assays["to_m"].astype(float) * to_m
    bad = ~(assays["to_m"] > assays["from_m"])
    if bad.any():
        log.warning("dropping %d assay intervals with to <= from", int(bad.sum()))
        assays = assays[~bad]
    n_bdl = int((pd.to_numeric(assays["cu"], errors="coerce") < 0).sum())
    assays["cu"] = clean_cu(assays["cu"], cfg.cu_unit or "pct", cfg.below_detection_factor)

    known = set(collars["hole_id"])
    for name, df in (("surveys", surveys), ("assays", assays)):
        orphans = ~df["hole_id"].isin(known)
        if orphans.any():
            log.warning("%s: dropping %d rows for holes without collars", name, int(orphans.sum()))
    surveys = surveys[surveys["hole_id"].isin(known)]
    assays = assays[assays["hole_id"].isin(known)]

    # Collar depth and orientation: fall back to deepest record / first survey / vertical.
    deepest = pd.concat(
        [assays.groupby("hole_id")["to_m"].max(), surveys.groupby("hole_id")["depth"].max()],
        axis=1,
    ).max(axis=1)
    if "total_depth" in collars:
        collars["total_depth"] = collars["total_depth"].astype(float) * to_m
        collars["total_depth"] = collars["total_depth"].fillna(collars["hole_id"].map(deepest))
    else:
        collars["total_depth"] = collars["hole_id"].map(deepest)
    first = surveys.sort_values("depth").groupby("hole_id").first()
    for col, default in (("azimuth", 0.0), ("dip", -90.0)):
        from_survey = collars["hole_id"].map(first[col])
        if col in collars:
            values = collars[col].astype(float) * (dip_sign if col == "dip" else 1.0)
            collars[col] = values.fillna(from_survey)
        else:
            collars[col] = from_survey
        n_default = int(collars[col].isna().sum())
        if n_default:
            log.warning("collars: %d holes have no %s; assuming %s", n_default, col, default)
        collars[col] = collars[col].fillna(default)
    no_depth = collars["total_depth"].isna()
    if no_depth.any():
        log.warning("collars: dropping %d holes with no depth information", int(no_depth.sum()))
        collars = collars[~no_depth]

    log.info(
        "loaded holes=%d surveys=%d assays=%d below_detection=%d not_analysed=%d origin=%s",
        len(collars),
        len(surveys),
        len(assays),
        n_bdl,
        int(assays["cu"].isna().sum()),
        origin,
    )
    return DrillholeTables(
        collars.reset_index(drop=True),
        surveys.reset_index(drop=True),
        assays.reset_index(drop=True),
        origin,
    )
