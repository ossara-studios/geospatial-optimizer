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

### NVIDIA GPU (Windows or Linux)
The ore model uses the GPU automatically when PyTorch can see one (`model.device: auto`), in
float32; otherwise it runs on the CPU in float64. The two agree on 99.97 % of ore calls.
PyPI's Windows torch wheel is CPU-only, so after the steps above install the CUDA build of the
same version. torch 2.15.0 ships `cu132` and `cu134`; `cu132` runs on any driver that reports
CUDA 13.x in `nvidia-smi` (tested on 581.42, CUDA 13.0):
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.lock
.venv\Scripts\pip install --force-reinstall --no-deps torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\pip install --no-deps -e .
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The model logs `device=cuda` on each fit. Raw data is git-ignored: copy `data/raw/hillside/`
(11 MB) from a machine that has it, then run `geoopt prepare`.

### Backtest
```bash
.venv/bin/geoopt backtest --workers 3
.venv/bin/geoopt backtest --beta-sweep --workers 3
.venv/bin/geoopt backtest --boundary-sweep --workers 3
```
`--workers` runs independent runs in parallel processes. Results match a serial run exactly.
On one 8 GB GPU, 2–3 workers is a sensible start.

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

## Live demo (animated)
Four terminals, in this order. The live page only reads OPC UA and the run folder, so it works
unchanged against real EAE:
```bash
.venv/bin/geoopt baseline                   # once: near-ore rule of thumb, for "cost saved"
.venv/bin/geoopt emulator --auto-approve    # stands in for EAE (drop --auto-approve to approve by hand)
.venv/bin/geoopt rig-sim
.venv/bin/geoopt live                       # http://localhost:8600
.venv/bin/geoopt run-loop --bridge opcua --max-holes 20
```
`http://localhost:8600/?replay` replays the latest finished run without OPC UA (`&speed=2` for
faster; `&run=<folder>` for another run). The KPIs are INDICATIVE: new ore uses an assumed
2.8 t/m³ and drilling an assumed A$373.70/m (see `config/default.yaml` → `kpi`). The planner
also writes them to the `Kpi.*` OPC UA variables for the EAE HMI.

## Visualizer
```bash
.venv/bin/geoopt viz
```
Opens on http://localhost:8501. The Run tab shows any run in `runs/` (turn on live refresh during
an OPC UA run); the Backtest tab shows `runs/backtest` results.

## Checks
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q
```
