# geospatial-optimizer

Recommends the next copper exploration drillhole, checks it against site limits, "drills" it
against held-out historical assays, updates the ore model and repeats.

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

## Live demo (animated)
One command, one process: the planner runs at demo pace and the page follows it.
```bash
.venv/bin/geoopt baseline       # once: near-ore rule of thumb, for "cost saved"
.venv/bin/geoopt demo           # http://localhost:8600
```
`--max-holes 40` runs longer; `--speed 120` drills faster (metres per second, default
`rig.metres_per_second`). `geoopt live` serves finished runs only:
`http://localhost:8600/?replay` replays the latest (`&speed=2` for faster; `&run=<folder>` for
another run). The KPIs are INDICATIVE: new ore uses an assumed 2.8 t/m³ and drilling an
assumed A$373.70/m (see `config/default.yaml` → `kpi`).

## Visualizer
```bash
.venv/bin/geoopt viz
```
Opens on http://localhost:8501. The Run tab shows any run in `runs/` (turn on live refresh during
a run); the Backtest tab shows `runs/backtest` results.

## Checks
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q
```
