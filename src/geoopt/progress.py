"""Where a demo run is in the drilling cycle, for the live view. Plain data, no logic.

The decision loop and the paced in-process bridge write it; the live page reads it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Cycle stages, in order (the live page names them).
STAGES = ("recommend", "check", "approve", "drill", "assays", "learn")


@dataclass
class Progress:
    running: bool = False
    req_id: int = 0
    hole_idx: int | None = None
    stage: int = -1  # index into STAGES; -1 = not started
    rejected: bool = False  # the current request failed the limit check
    depth_m: float = 0.0  # metres drilled so far on the current hole
    target_m: float = 0.0  # metres this hole will reach
    pause_s: float = 0.5  # how long each instant stage stays on screen

    def at(self, stage: str, **changes: object) -> None:
        self.stage = STAGES.index(stage)
        for k, v in changes.items():
            setattr(self, k, v)

    def as_dict(self) -> dict:
        return asdict(self)
