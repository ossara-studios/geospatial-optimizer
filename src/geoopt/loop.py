"""The bridge-agnostic decision loop (state machine in docs/eae_contract.md).

recommend -> check -> approve -> dispatch -> complete -> ingest lab results -> condition.
With `assay_lag_holes` = k, a hole's results condition the model only after k more holes.
The loop only sees grades through the lab drop folder; it never touches held-out data.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from geoopt.bridge.base import Bridge
from geoopt.config import LoopConfig
from geoopt.kpi import DiscoveryTracker
from geoopt.model.gp import OreModel
from geoopt.planner.strategies import PlannerState, Strategy
from geoopt.protocol import CheckStatus, Decision, RecommendationMsg

log = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    iteration: int
    req_id: int
    hole_idx: int
    hole_id: str
    score: float
    exp_ore_m: float
    check_status: str
    check_reason: str
    checked_depth: float
    decision: str | None = None
    metres_drilled: float = 0.0
    composites_ingested: int = 0
    cumulative_metres: float = 0.0
    prob_ore: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    kpis: dict[str, float] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)


def eval_grid(points: pd.DataFrame, shape: tuple[int, int, int]) -> np.ndarray:
    """Regular grid over the candidate sample extent, for visualization."""
    lo = points[["x", "y", "z"]].min().to_numpy()
    hi = points[["x", "y", "z"]].max().to_numpy()
    axes = [np.linspace(lo[i], hi[i], shape[i]) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class DecisionLoop:
    def __init__(
        self,
        cfg: LoopConfig,
        strategy: Strategy,
        bridge: Bridge,
        model: OreModel,
        candidates: pd.DataFrame,
        points: pd.DataFrame,
        seed_collars_xy: np.ndarray,
        run_dir: Path,
        lab_dir: Path,
        write_grid: bool = True,
        on_iteration: Callable[[IterationRecord], None] | None = None,
        kpi: DiscoveryTracker | None = None,
    ) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._bridge = bridge
        self._model = model
        self._cand = candidates.set_index("hole_idx", drop=False)
        self._run_dir = run_dir
        self._lab_dir = lab_dir
        self._grid = eval_grid(points, cfg.eval_grid) if write_grid else None
        self.on_iteration = on_iteration
        self._kpi = kpi
        self.state = PlannerState(
            candidates=candidates,
            points=points,
            available=set(candidates["hole_idx"].astype(int)),
            drilled_xy=np.asarray(seed_collars_xy, dtype=float).reshape(-1, 2),
            model=model,
        )
        self.records: list[IterationRecord] = []
        # Drilled holes whose assays the lab hasn't returned yet (oldest first).
        self._pending: deque[pd.DataFrame] = deque()
        self._req_id = 0
        self._metres = 0.0

    @property
    def model(self) -> OreModel:
        return self._model

    @property
    def drilled_holes(self) -> list[int]:
        return [r.hole_idx for r in self.records if r.metres_drilled > 0]

    async def run(self) -> list[IterationRecord]:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._req_id = await self._bridge.next_req_id() - 1
        drilled = 0
        iteration = 0
        while drilled < self._cfg.max_holes and self.state.available:
            if self._cfg.max_metres is not None and self._metres >= self._cfg.max_metres:
                break
            iteration += 1
            record = await self._step(iteration)
            self.records.append(record)
            self._write(record)
            if self.on_iteration:
                self.on_iteration(record)
            if record.metres_drilled > 0:
                drilled += 1
        log.info("loop done iterations=%d holes=%d metres=%.0f", iteration, drilled, self._metres)
        return self.records

    async def _step(self, iteration: int) -> IterationRecord:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        rec = self._strategy.recommend(self.state)
        t["recommend"] = time.perf_counter() - t0
        row = self._cand.loc[rec.hole_idx]
        self._req_id += 1
        msg = RecommendationMsg(
            req_id=self._req_id,
            hole_idx=rec.hole_idx,
            collar_x=float(row["x"]),
            collar_y=float(row["y"]),
            collar_z=float(row["z"]),
            azimuth=float(row["azimuth"]),
            dip=float(row["dip"]),
            depth=float(row["depth"]),
            prob_ore=_finite(rec.prob_ore),
            exp_ore_m=_finite(rec.exp_ore_m),
            score=_finite(rec.score),
        )

        t1 = time.perf_counter()
        check = await self._bridge.submit(msg)
        t["check"] = time.perf_counter() - t1
        record = IterationRecord(
            iteration=iteration,
            req_id=msg.req_id,
            hole_idx=msg.hole_idx,
            hole_id=str(row["hole_id"]),
            score=msg.score,
            exp_ore_m=msg.exp_ore_m,
            check_status=check.status.name,
            check_reason=check.reason.name,
            checked_depth=check.depth,
            cumulative_metres=self._metres,
            prob_ore=msg.prob_ore,
            breakdown=rec.breakdown,
            timings_s=t,
        )
        if check.status is CheckStatus.REJECTED:
            self.state.available.discard(msg.hole_idx)
            self._log(record)
            return record

        t2 = time.perf_counter()
        decision = await self._bridge.await_approval(msg.req_id)
        t["approval"] = time.perf_counter() - t2
        record.decision = decision.name
        if decision is Decision.DENIED:
            self.state.available.discard(msg.hole_idx)
            self._log(record)
            return record

        t3 = time.perf_counter()
        done = await self._bridge.await_completion(msg.req_id)
        t["drill"] = time.perf_counter() - t3
        if done.req_id != msg.req_id or done.hole_idx != msg.hole_idx:
            raise RuntimeError(f"completion {done} does not match request {msg.req_id}")

        t4 = time.perf_counter()
        lab = pd.read_parquet(self._lab_dir / f"{msg.hole_idx}.parquet")
        self._pending.append(lab)
        if self._kpi is not None:
            self._kpi.add_hole(done.metres_drilled, None)
        # With a lag of k, a hole's assays arrive once k more holes have been drilled.
        while len(self._pending) > self._cfg.assay_lag_holes:
            ready = self._pending.popleft()
            if len(ready):
                self._model.condition(ready[["x", "y", "z"]].to_numpy(), ready["cu"].to_numpy())
                if self._kpi is not None:
                    self._kpi.add_assays(ready)
        t["condition"] = time.perf_counter() - t4
        if self._kpi is not None:
            record.kpis = self._kpi.kpis().as_dict()
            await self._bridge.publish_kpis(record.kpis)

        self.state.available.discard(msg.hole_idx)
        self.state.drilled_xy = np.vstack([self.state.drilled_xy, [[msg.collar_x, msg.collar_y]]])
        self._metres += done.metres_drilled
        record.metres_drilled = done.metres_drilled
        record.composites_ingested = len(lab)
        record.cumulative_metres = self._metres
        self._log(record)
        return record

    def _log(self, r: IterationRecord) -> None:
        log.info(
            "iter=%d req_id=%d hole_idx=%d hole=%s score=%.3f check=%s/%s decision=%s "
            "metres=%.1f cum_metres=%.0f timings=%s",
            r.iteration,
            r.req_id,
            r.hole_idx,
            r.hole_id,
            r.score,
            r.check_status,
            r.check_reason,
            r.decision,
            r.metres_drilled,
            r.cumulative_metres,
            {k: round(v, 2) for k, v in r.timings_s.items()},
        )

    def _write(self, record: IterationRecord) -> None:
        it_dir = self._run_dir / f"iter_{record.iteration:03d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        (it_dir / "record.json").write_text(json.dumps(asdict(record), indent=2))
        drilled = self._cand[~self._cand["hole_idx"].isin(self.state.available)]
        drilled[["hole_idx", "hole_id", "x", "y", "z", "azimuth", "dip", "depth"]].to_parquet(
            it_dir / "excluded_or_drilled.parquet", index=False
        )
        if self._grid is not None:
            mean, std = self._model.predict(self._grid)
            pd.DataFrame(
                {
                    "x": self._grid[:, 0],
                    "y": self._grid[:, 1],
                    "z": self._grid[:, 2],
                    "mean_log": mean,
                    "std_log": std,
                    "p_ore": self._model.prob_ore(self._grid),
                }
            ).to_parquet(it_dir / "grid.parquet", index=False)


def _finite(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0
