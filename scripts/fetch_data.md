# The Hillside dataset

**Source:** Department for Energy and Mining, Government of South Australia, Geoscientific Data
(SARIG data package, July 2025 extract), sourced on 4 November 2026,
https://dem-sdp.s3-ap-southeast-2.amazonaws.com/index.html. Licensed CC-BY 4.0.
The raw files stay in `data/raw/` (git-ignored); attribution goes in the README and docs.

**Deposit:** Hillside Cu-Au (Rex Minerals (SA) Pty Ltd), Yorke Peninsula, South Australia. That's
the Rex Minerals mineral-exploration holes on the Stansbury 1:100k map sheet, drilled 2006–2013,
in GDA2020 MGA zone 53 (metres).

## Getting it
```bash
python scripts/fetch_hillside.py
```
Use the venv's Python (see the README). The script does three steps:

1. **Drillhole details (140 MB):** reads `sarig_dh_details_exp.csv` and keeps the rows where
   `OPERATOR_NAME` contains "Rex Minerals", `MAP_100000` contains "Stansbury" and
   `MINERAL_CLASS == "Y"` (993 holes). Writes `data/raw/sarig/hillside_collars.csv`.
2. **Geochemistry (23.4 GB, not stored):** streams `sarig_rs_chem_exp.csv` over parallel HTTP
   byte ranges and keeps the rows where `CHEM_CODE == "Cu"` and `DRILLHOLE_NUMBER` is one of
   step 1's holes. Writes `data/raw/sarig/hillside_cu.csv`. Finished byte ranges are kept in
   `data/raw/sarig/cu_parts/` until the end, so an interrupted run resumes.
3. **Canonical tables:** runs `scripts/build_hillside.py`, which writes
   `data/raw/hillside/{collars,surveys,assays}.csv` and `build_report.json`. The report records
   every exclusion (aircore holes, holes outside the deposit area) and every fill (missing collar
   elevations taken from neighbouring collars).

After that, `geoopt prepare` (or the first `geoopt demo`) writes the processed tables to
`data/processed/`.

`sarig_rs_details_exp.csv` (1.2 GB) isn't needed. It was used only to check sample coverage
before the chem file was first fetched.
