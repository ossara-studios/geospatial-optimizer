"""Download the Hillside drillhole data from SARIG and build data/raw/hillside/.

    python scripts/fetch_hillside.py            # ~20 min on a fast connection
    python scripts/fetch_hillside.py --workers 48

Steps:
1. Drillhole details (140 MB): keep the Rex Minerals mineral holes on the Stansbury 1:100k sheet
   -> data/raw/sarig/hillside_collars.csv.
2. Geochemistry (23.4 GB): stream it over parallel byte ranges (the bucket serves about
   0.6 MB/s per connection) and keep only those holes' Cu rows -> data/raw/sarig/hillside_cu.csv.
   Nothing but the kept rows is written to disk. An interrupted run resumes: finished parts
   are skipped.
3. Run scripts/build_hillside.py -> data/raw/hillside/{collars,surveys,assays}.csv.

Source: Department for Energy and Mining, Government of South Australia, Geoscientific Data
(SARIG data package, July 2025 extract), CC-BY 4.0, https://map.sarig.sa.gov.au/
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://dem-sdp.s3-ap-southeast-2.amazonaws.com"
DETAILS_URL = f"{BUCKET}/sarig_dh_details_exp.csv"
CHEM_URL = f"{BUCKET}/sarig_rs_chem_exp.csv"
OVERSHOOT = 1 << 20  # read 1 MiB past a part's end to finish its last line
RETRIES = 5
REPO = Path(__file__).resolve().parents[1]


def _open(url: str, start: int | None = None, end: int | None = None):  # noqa: ANN202
    req = urllib.request.Request(url, headers={"User-Agent": "geoopt-fetch"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req, timeout=60)


def _size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "geoopt-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_collars(out: Path) -> set[str]:
    """Stream the details file, keep the Hillside holes, return their SARIG hole numbers."""
    print("1/3 drillhole details (140 MB) ...", flush=True)
    with _open(DETAILS_URL) as r:
        reader = csv.DictReader(io.TextIOWrapper(r, encoding="utf-8", errors="replace", newline=""))
        rows = [
            row
            for row in reader
            if "Rex Minerals" in (row["OPERATOR_NAME"] or "")
            and "Stansbury" in (row["MAP_100000"] or "")
            and row["MINERAL_CLASS"] == "Y"
        ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames or [], quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)
    print(f"    kept {len(rows)} holes -> {out}", flush=True)
    return {row["DRILLHOLE_NO"] for row in rows}


def fetch_cu(holes: set[str], out: Path, parts_dir: Path, workers: int) -> int:
    size = _size(CHEM_URL)
    print(f"2/3 geochemistry ({size / 1e9:.1f} GB, streamed, not stored) ...", flush=True)
    with _open(CHEM_URL, 0, 4095) as r:
        header = r.read().split(b"\n", 1)[0].decode().strip()
    columns = next(csv.reader([header]))
    i_hole, i_code = columns.index("DRILLHOLE_NUMBER"), columns.index("CHEM_CODE")
    needle = b',"Cu",'  # cheap pre-filter before parsing a line
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_parts = workers * 4  # smaller parts: less to redo after an interruption
    done_bytes = [0] * n_parts
    kept = [0] * n_parts
    failed: list[str] = []

    def part(k: int) -> None:
        start, end = k * size // n_parts, (k + 1) * size // n_parts - 1
        path, done = parts_dir / f"part_{k:04d}.csv", parts_dir / f"part_{k:04d}.done"
        if done.exists():
            done_bytes[k] = end - start + 1
            return
        for attempt in range(RETRIES):
            try:
                pos, first, n = start, start > 0, 0
                with (
                    _open(CHEM_URL, start, min(size - 1, end + OVERSHOOT)) as r,
                    path.open("w", newline="", encoding="utf-8") as f,
                ):
                    w = csv.writer(f)
                    for line in r:
                        line_start = pos
                        pos += len(line)
                        done_bytes[k] = min(pos, end + 1) - start
                        if first:  # the previous part owns this partial line
                            first = False
                            continue
                        if line_start > end:
                            break
                        if needle in line:
                            row = next(csv.reader([line.decode("utf-8", "replace")]))
                            if len(row) > i_code and row[i_code] == "Cu" and row[i_hole] in holes:
                                w.writerow(row)
                                n += 1
                kept[k] = n
                done.touch()
                return
            except OSError as exc:
                if attempt == RETRIES - 1:
                    failed.append(f"part {k}: {exc}")
                time.sleep(2 * (attempt + 1))

    stop = threading.Event()

    def report() -> None:
        t0, base = time.time(), sum(done_bytes)
        while not stop.wait(15):
            got = sum(done_bytes)
            rate = (got - base) / max(time.time() - t0, 1e-9)
            eta = (size - got) / rate / 60 if rate > 0 else float("inf")
            print(f"    {got / 1e9:5.1f} / {size / 1e9:.1f} GB  {rate / 1e6:5.1f} MB/s  "
                  f"about {eta:4.0f} min left", flush=True)  # fmt: skip

    threading.Thread(target=report, daemon=True).start()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(part, range(n_parts)))
    stop.set()
    if failed:
        sys.exit("some parts failed (run again to resume):\n  " + "\n  ".join(failed))
    total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)
        for k in range(n_parts):
            text = (parts_dir / f"part_{k:04d}.csv").read_text(encoding="utf-8")
            f.write(text)
            total += text.count("\n")
    shutil.rmtree(parts_dir)
    print(f"    kept {total:,} Cu rows -> {out}", flush=True)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=32, help="parallel connections (default 32)")
    p.add_argument("--sarig-dir", type=Path, default=REPO / "data/raw/sarig")
    p.add_argument("--out-dir", type=Path, default=REPO / "data/raw/hillside")
    a = p.parse_args()
    a.sarig_dir.mkdir(parents=True, exist_ok=True)
    holes = fetch_collars(a.sarig_dir / "hillside_collars.csv")
    fetch_cu(holes, a.sarig_dir / "hillside_cu.csv", a.sarig_dir / "cu_parts", a.workers)
    print("3/3 building the canonical tables ...", flush=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts/build_hillside.py"),
         "--sarig-dir", str(a.sarig_dir), "--out-dir", str(a.out_dir)],
        check=True,
    )  # fmt: skip
    print(f"done: {a.out_dir}. Next: geoopt demo (it prepares the data on first run).")


if __name__ == "__main__":
    main()
