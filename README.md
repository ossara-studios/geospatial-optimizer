# geospatial-optimizer

Recommends the next copper exploration drillhole and runs each recommendation through an
EcoStruxure Automation Expert (IEC 61499) loop over OPC UA: limit checks, operator approval, dispatch.

## Data
Hillside Cu-Au deposit (Rex Minerals), from the Geological Survey of South Australia's SARIG data
package. Attribution: Department for Energy and Mining, Government of South Australia,
Geoscientific Data, sourced on 4 November 2026, https://map.sarig.sa.gov.au/ (CC-BY 4.0).
To rebuild it, see [scripts/fetch_data.md](scripts/fetch_data.md). The raw data isn't committed.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
```

## OPC UA round-trip (Phase 0)
In one terminal, start the EAE emulator (it prompts y/n for approval; add `--auto-approve` to skip the prompt):
```bash
.venv/bin/geoopt emulator
```
In another, send one recommendation:
```bash
.venv/bin/python scripts/opcua_roundtrip.py --depth 650
```
To test against real EAE, pass `--endpoint opc.tcp://<eae-host>:<port>`. See [docs/eae_contract.md](docs/eae_contract.md).

## Checks
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q
```
