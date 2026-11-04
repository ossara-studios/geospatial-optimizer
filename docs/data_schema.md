# Data schema

## Source: Hillside Cu-Au deposit, from SARIG (Geological Survey of South Australia)
**Attribution (CC-BY 4.0):** Department for Energy and Mining, Government of South Australia,
Geoscientific Data (SARIG data package, July 2025 extract), sourced on 4 November 2026,
https://map.sarig.sa.gov.au/

**Deposit:** Hillside (Rex Minerals (SA) Pty Ltd), Yorke Peninsula, SA. That's the Rex Minerals
mineral-exploration holes on the Stansbury 1:100k sheet, drilled 2006–2013. How the files are
fetched and built is in [scripts/fetch_data.md](../scripts/fetch_data.md).

Copper Creek (Faraday Copper) was the original choice, but its download page is offline. Its
notes are kept below in case it comes back.

### Raw SARIG tables used
| SARIG file | Columns used | Notes |
|---|---|---|
| `sarig_dh_details_exp.csv` | `DRILLHOLE_NO`, `DH_NAME`, `EASTING/NORTHING_GDA2020`, `ZONE_GDA2020`, `ELEVATION_M`, `INCLINATION`, `AZIMUTH`, `MAX_DRILLED_DEPTH`, `MAX_DRILLED_DEPTH_DATE`, `DRILLING_METHODS` | One row per hole. Collars are GDA2020 / MGA zone 53 (metres). `INCLINATION` is negative down. Every Hillside hole has a date. |
| `sarig_rs_chem_exp.csv` | `SAMPLE_NO`, `DRILLHOLE_NUMBER`, `DH_DEPTH_FROM/TO`, `CHEM_CODE`, `VALUE`, `UNIT`, `CHEM_METHOD_CODE`, `LABORATORY`, `SAMPLE_ANALYSIS_NO` | Long format (one row per sample per element). Hillside Cu is in **ppm**, mostly 1 m samples. |

### Build decisions (`scripts/build_hillside.py`; counts go in `build_report.json`)
- **Aircore holes are excluded** (shallow, low sample quality). We keep diamond core and RC.
- **There are no downhole surveys in SARIG.** Each hole gets one station from its collar
  azimuth/dip, so holes are modelled as straight lines. This is a stated limitation.
- **Missing collar elevations** (about 20% of holes) are filled with the median of the 5 nearest
  surveyed collars. The terrain is flat, 23–157 m.
- **Cu:** ppm is converted to %. Where a sample has several analyses, fused-bead XRF
  (`ME-XRF21n`, ALS) is preferred over ICP-MS (`IC3M`, AMDEL). Overlapping intervals in a hole
  keep the first. Negative values (−DL) pass through to the loader.

### Copper Creek notes (not in use)
- Faraday's notes say the data is an Excel workbook in WGS 84 UTM 12N metres.
- A blank assay cell means not analysed, and below-detection is stored as **−DL**.

## Canonical tables (what `geoopt.data.load` produces)
All lengths are in metres. X is east, Y north, Z up, in **local** coordinates (source CRS minus `origin`; the origin is written to `meta.json`).

| Table | Columns |
|---|---|
| collars | `hole_id`, `x`, `y`, `z`, `total_depth`, `azimuth`, `dip`, `drill_date` (optional) |
| surveys | `hole_id`, `depth`, `azimuth`, `dip` |
| assays | `hole_id`, `from_m`, `to_m`, `cu` (% Cu; NaN = not analysed) |

Conventions:
- **Azimuth** is in degrees clockwise from grid north. **Dip** is in degrees, **negative = downward** (−90 = vertical down). Raw positive-down dips are flipped.
- **Below detection** (−DL) becomes `DL × below_detection_factor` (default 0.5).
- **Not analysed** (blank) stays NaN. It carries no weight in compositing and is never zero-filled.
- Holes without surveys use the collar azimuth/dip. If there are none, the hole is assumed vertical, and a warning reports how many.

## Processed outputs (`data/processed/`, git-ignored)
| File | Contents | Who may read it |
|---|---|---|
| `seed_composites.parquet` | composites of the seed holes, with `cu` and x/y/z | planner, model |
| `candidates.parquet` | one row per held-out hole: `hole_idx`, `hole_id`, collar, `azimuth`, `dip`, `depth` | planner |
| `candidate_points.parquet` | regular bins along each candidate's full depth, **geometry only, no grades** | planner |
| `collars.parquet` | all holes' collar geometry | planner, viz |
| `meta.json` | origin, counts, seed hole ids, settings | anyone |
| `truth/heldout_composites.parquet` | held-out composites **with grades** | **rig/lab simulator only** |

Candidate points come from the hole geometry, not from the assay intervals. Otherwise, *which* intervals were assayed (geologists sample what looks mineralised) would leak into the planner.

## Compositing
Fixed-length bins of `composite.length_m`, measured from the collar, with a length-weighted mean grade. A bin is kept when its assayed length is at least `min_coverage` × its length.
