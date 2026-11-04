"""Codes and messages shared by the planner, the bridges and EAE (see docs/eae_contract.md)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CheckStatus(IntEnum):
    ACCEPTED = 1
    CLIPPED = 2
    REJECTED = 3


class Reason(IntEnum):
    NONE = 0
    OUT_OF_BOUNDS = 1
    NO_GO_ZONE = 2
    DEPTH_CLIPPED = 3
    DIP_OUT_OF_RANGE = 4
    AZIMUTH_OUT_OF_RANGE = 5


class Decision(IntEnum):
    APPROVED = 1
    DENIED = 2


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    reason: Reason
    depth: float
