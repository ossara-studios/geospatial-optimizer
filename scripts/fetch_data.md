# Fetching the Hillside dataset

**Source:** Department for Energy and Mining, Government of South Australia, Geoscientific Data
(SARIG data package, July 2025 extract), sourced on 4 November 2026,
https://dem-sdp.s3-ap-southeast-2.amazonaws.com/index.html. Licensed CC-BY 4.0.
The raw files stay in `data/raw/` (git-ignored); attribution goes in the README and docs.

**Deposit:** Hillside Cu-Au (Rex Minerals (SA) Pty Ltd), Yorke Peninsula, South Australia. That's
the Rex Minerals mineral-exploration holes on the Stansbury 1:100k map sheet, drilled 2006–2013,
in GDA2020 MGA zone 53 (metres).

## Steps
1. **Drillhole details (140 MB):** download `sarig_dh_details_exp.csv` into `data/raw/sarig/`.
   Keep the rows where `OPERATOR_NAME` contains "Rex Minerals", `MAP_100000` contains
   "Stansbury" and `MINERAL_CLASS == "Y"`. Write them to `data/raw/sarig/hillside_collars.csv`
   (993 holes).
2. **Geochemistry (23.4 GB, not stored):** fetch `sarig_rs_chem_exp.csv` over parallel HTTP byte
   ranges, since the bucket serves about 0.6 MB/s per connection. Keep the rows where
   `CHEM_CODE == "Cu"` and `DRILLHOLE_NUMBER` is in step 1's hole list, and write them to
   `data/raw/sarig/hillside_cu.csv`.
3. **Canonical tables:**
   ```bash
   .venv/bin/python scripts/build_hillside.py
   ```
   This writes `data/raw/hillside/{collars,surveys,assays}.csv` and `build_report.json`, which
   records every exclusion and fill.
4. **Processed outputs:**
   ```bash
   .venv/bin/geoopt prepare
   ```

`sarig_rs_details_exp.csv` (1.2 GB) is optional. We used it only to check sample coverage before
fetching the chem file.
