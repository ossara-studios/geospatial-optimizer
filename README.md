# geospatial-optimizer

An adaptive drill planner for copper exploration. It recommends the next hole, checks it against
site limits, "drills" it by releasing that hole's held-out historical assays, updates a Gaussian
process ore model and repeats. The planner never sees an assay before its hole is drilled.

Backtest results, including the comparison with drilling beside known ore, are in
[docs/results.md](docs/results.md).

## Setup
Python 3.11 or newer (`requirements.lock` was frozen on 3.13).

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.lock
.venv\Scripts\pip install --no-deps -e .
```
macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
```
The raw data isn't in the repo (see [Data](#data)). Copy `data/raw/hillside/` (about 12 MB) from
a machine that has it. The demo prepares it on first run; otherwise run `geoopt prepare`.

The commands below are written as `geoopt …`. Run them as `.venv\Scripts\geoopt …` on Windows
or `.venv/bin/geoopt …` on macOS/Linux, or activate the venv first.

### NVIDIA GPU (optional)
The ore model uses the GPU when PyTorch can see one (`model.device: auto`), in float32;
otherwise it runs on the CPU in float64. PyPI's Windows torch wheel is CPU-only, so after the
setup above install the CUDA build of the same version (tested: RTX 4060 Laptop, driver 581.42):
```powershell
.venv\Scripts\pip install --force-reinstall --no-deps torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The model logs `device=cuda` on each fit.

## Live demo
```bash
geoopt demo
```
This opens http://localhost:8600 and drills 20 holes at demo pace, showing each stage in turn:
recommendation, limit check, approval (automatic), drilling, assays, model update. A hole takes
about 4–11 s depending on its depth. The first run also prepares the data and builds the
rule-of-thumb baseline, which takes under a minute. When the run ends, the page stays up until
you press Ctrl+C.

| Option | Does |
|---|---|
| `--holes 40` | drill more holes (default `loop.max_holes`, 20) |
| `--speed 120` | drilling speed in metres per second (default `rig.metres_per_second`, 60) |
| `--port 8700` | use another port (default 8600) |
| `--no-open` | don't open the browser |

`geoopt live` serves finished runs only. `http://localhost:8600/?replay` replays the latest one;
add `&speed=2` to go faster or `&run=<folder name>` to pick another run from `runs/`.

**What the figures mean.** "New ore found" counts 25 × 25 × 10 m blocks that an assayed ore
sample (Cu > 0.2 %) reaches for the first time. "Saved against the rule of thumb" compares the
metres the planner drilled with the metres a simple rule needs to find the same new ore. The rule
always drills the hole with the most of its length within 50 m of known ore. Tonnes use an
assumed 2.8 t/m³ and cost an assumed A$373.70/m (`config/default.yaml` → `kpi`), so treat both
as indicative. Over a 20-hole demo the planner is level with the rule or up to about 10 % ahead. The gap opens
with more metres drilled.

## Other commands
| Command | Does |
|---|---|
| `geoopt prepare` | load the raw drillhole data and write the seed, candidate and held-out tables |
| `geoopt run-loop` | one run at full speed, no page (`--strategy adaptive\|random\|grid\|historical`, `--max-holes`) |
| `geoopt baseline` | rebuild the rule-of-thumb curve (the demo does this on first run) |
| `geoopt backtest` | every strategy over every seed set at the same metre budget |
| `geoopt viz` | Streamlit visualizer on http://localhost:8501: the Run tab shows any run in `runs/` (with optional 2 s refresh); the Backtest tab shows `runs/backtest` |

Backtest variants: `--beta-sweep`, `--boundary-sweep`, `--delineation-sweep` and
`--assay-delay-sweep`, each reading its list from `config/default.yaml` → `backtest`.
`--workers N` runs independent runs in parallel processes. On one 8 GB GPU, 2–3 workers is a
sensible start.

## Data
Hillside Cu-Au deposit (Rex Minerals), Yorke Peninsula, South Australia. The holes come from the
SARIG data package (July 2025 extract). Attribution: Department for Energy and Mining,
Government of South Australia, Geoscientific Data, sourced on 4 November 2026,
https://map.sarig.sa.gov.au/ (CC-BY 4.0). To rebuild the raw tables from SARIG, see
[scripts/fetch_data.md](scripts/fetch_data.md).

## Checks
```bash
ruff check . && ruff format --check . && pytest -q
```
Run inside the activated venv, or prefix each tool with the venv path as above.
