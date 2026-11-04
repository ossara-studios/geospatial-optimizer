"""Build canonical collars/surveys/assays CSVs for the Hillside Cu-Au deposit from SARIG.

Inputs (see scripts/fetch_data.md for how they are produced):
    data/raw/sarig/hillside_collars.csv   Rex Minerals holes on the Stansbury 1:100k sheet
    data/raw/sarig/hillside_cu.csv        their Cu rows from sarig_rs_chem_exp.csv

Outputs in data/raw/hillside/: collars.csv, surveys.csv, assays.csv, build_report.json

Source: Department for Energy and Mining, Government of South Australia, Geoscientific
Data (SARIG data package), CC-BY 4.0, https://map.sarig.sa.gov.au/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger("build_hillside")

EXCLUDED_METHODS = ("Aircore",)  # shallow, low-quality samples
ELEVATION_NEIGHBOURS = 5


def build_collars(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep = ~raw["DRILLING_METHODS"].fillna("").str.contains("|".join(EXCLUDED_METHODS))
    c = raw[keep].copy()
    report = {"holes_in": len(raw), "holes_excluded_aircore": int((~keep).sum())}

    # Fill missing collar elevations from the nearest surveyed collars (flat terrain).
    missing = c["ELEVATION_M"].isna().to_numpy()
    if missing.any():
        known = c.loc[~missing, ["EASTING_GDA2020", "NORTHING_GDA2020"]].to_numpy()
        tree = cKDTree(known)
        _, idx = tree.query(
            c.loc[missing, ["EASTING_GDA2020", "NORTHING_GDA2020"]].to_numpy(),
            k=ELEVATION_NEIGHBOURS,
        )
        elev = c.loc[~missing, "ELEVATION_M"].to_numpy()
        c.loc[missing, "ELEVATION_M"] = np.median(elev[idx], axis=1)
    report["elevation_filled_from_neighbours"] = int(missing.sum())

    collars = pd.DataFrame(
        {
            "hole_id": c["DH_NAME"].str.strip(),
            "sarig_drillhole_no": c["DRILLHOLE_NO"],
            "x": c["EASTING_GDA2020"],
            "y": c["NORTHING_GDA2020"],
            "z": c["ELEVATION_M"],
            "total_depth": c["MAX_DRILLED_DEPTH"],
            "azimuth": c["AZIMUTH"],
            "dip": c["INCLINATION"],
            "drill_date": pd.to_datetime(c["MAX_DRILLED_DEPTH_DATE"], dayfirst=True).dt.date,
            "method": c["DRILLING_METHODS"].str.strip(),
        }
    )
    report["holes_out"] = len(collars)
    return collars, report


def build_surveys(collars: pd.DataFrame) -> pd.DataFrame:
    """SARIG has no downhole surveys: one station per hole from the collar orientation."""
    return collars[["hole_id", "azimuth", "dip"]].assign(depth=0.0)[
        ["hole_id", "depth", "azimuth", "dip"]
    ]


PPM_PER_PCT = 10_000.0
# When a sample has several Cu analyses, prefer fused-bead XRF (total Cu) over the others.
METHOD_PREFERENCE = {"ME-XRF21n": 0, "IC3M": 1}


def build_assays(cu_rows: pd.DataFrame, collars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = cu_rows.copy()
    report: dict = {"cu_rows_in": len(df), "units": df["UNIT"].value_counts().to_dict()}
    unit_scale = {"ppm": 1.0 / PPM_PER_PCT, "%": 1.0, "pct": 1.0, "ppb": 1e-3 / PPM_PER_PCT}
    unknown = set(df["UNIT"]) - set(unit_scale)
    if unknown:
        raise ValueError(f"unhandled Cu units {unknown}")
    value = pd.to_numeric(df["VALUE"], errors="coerce")
    report["non_numeric_values_dropped"] = int(value.isna().sum())
    df["cu"] = value * df["UNIT"].map(unit_scale)
    # Negative values (-DL, below detection) pass through; the loader's
    # below_detection_factor converts them, so that rule lives in one place.
    report["below_detection_negative"] = int((df["cu"] < 0).sum())
    df = df.dropna(subset=["cu"])

    df["from_m"] = pd.to_numeric(df["DH_DEPTH_FROM"], errors="coerce")
    df["to_m"] = pd.to_numeric(df["DH_DEPTH_TO"], errors="coerce")
    bad = ~(df["to_m"] > df["from_m"])
    report["bad_intervals_dropped"] = int(bad.sum())
    df = df[~bad]

    df["pref"] = df["CHEM_METHOD_CODE"].map(METHOD_PREFERENCE).fillna(9)
    before = len(df)
    df = df.sort_values(["SAMPLE_NO", "pref", "SAMPLE_ANALYSIS_NO"]).drop_duplicates(
        "SAMPLE_NO", keep="first"
    )
    report["duplicate_analyses_dropped"] = before - len(df)

    id_map = dict(zip(collars["sarig_drillhole_no"].astype(str), collars["hole_id"], strict=True))
    df["hole_id"] = df["DRILLHOLE_NUMBER"].map(id_map)
    report["rows_for_excluded_holes_dropped"] = int(df["hole_id"].isna().sum())
    df = df.dropna(subset=["hole_id"])

    # Overlapping intervals within a hole (re-splits, field duplicates): keep the first.
    df = df.sort_values(["hole_id", "from_m", "to_m"])
    prev_to = df.groupby("hole_id")["to_m"].shift().fillna(-np.inf)
    overlap = df["from_m"] < prev_to - 1e-6
    report["overlapping_intervals_dropped"] = int(overlap.sum())
    df = df[~overlap]

    report["methods"] = df["CHEM_METHOD_CODE"].value_counts().to_dict()
    report["labs"] = df["LABORATORY"].value_counts().to_dict()
    report["assays_out"] = len(df)
    report["holes_with_assays"] = int(df["hole_id"].nunique())
    assays = df[["hole_id", "from_m", "to_m", "cu"]].assign(
        method=df["CHEM_METHOD_CODE"], lab=df["LABORATORY"]
    )
    return assays.reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sarig-dir", type=Path, default=Path("data/raw/sarig"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/hillside"))
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    collars, report = build_collars(pd.read_csv(args.sarig_dir / "hillside_collars.csv"))
    assays, assay_report = build_assays(
        pd.read_csv(args.sarig_dir / "hillside_cu.csv", dtype=str), collars
    )
    report |= assay_report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    collars.to_csv(args.out_dir / "collars.csv", index=False)
    build_surveys(collars).to_csv(args.out_dir / "surveys.csv", index=False)
    assays.to_csv(args.out_dir / "assays.csv", index=False)
    (args.out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    log.info("built %s", report)


if __name__ == "__main__":
    main()
