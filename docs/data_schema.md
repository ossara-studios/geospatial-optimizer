# Data schema

## Source: Faraday Copper, Copper Creek drillhole database
Status: **raw files not yet inspected.** The sections marked *TBD* are filled in once the files are in `data/raw/`.

What Faraday's *Notes to Drillhole Database* (May 2024) says about the files:
- They're in an Excel spreadsheet. Holes date from 1956 to Faraday's Phase III results (to May 14, 2024), and they include the database SRK used for the February 2023 resource estimate.
- Collar coordinates are in **WGS 84 UTM Zone 12N** (metres).
- Assays: Cu, Mo, Ag, Au. A **blank** cell means no analysis was done. A value **below the detection limit** is stored as **−DL**.

### Raw tables (TBD after inspection)
| Table | File / sheet | Columns → canonical | Notes |
|---|---|---|---|
| Collars | TBD | TBD | Drill dates present? |
| Surveys | TBD | TBD | Dip sign convention? |
| Assays | TBD | TBD | Cu unit (% or ppm)? Depth unit (m or ft)? |

Record these in the `data` section of `config/default.yaml`. `geoopt prepare` refuses to run while any of them is null.

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
