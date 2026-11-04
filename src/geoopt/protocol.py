"""Codes and messages shared by the planner, the bridges and EAE (see docs/eae_contract.md)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


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


class RigState(IntEnum):
    IDLE = 0
    DRILLING = 1
    COMPLETE = 2
    FAULT = 9


class Decision(IntEnum):
    APPROVED = 1
    DENIED = 2


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    reason: Reason
    depth: float


class RecommendationMsg(BaseModel):
    """Everything the planner sends to EAE for one request (all scalars)."""

    model_config = ConfigDict(frozen=True)

    req_id: int = Field(gt=0)
    hole_idx: int = Field(ge=0)
    collar_x: float
    collar_y: float
    collar_z: float
    azimuth: float
    dip: float
    depth: float = Field(gt=0)
    prob_ore: float
    exp_ore_m: float
    score: float


@dataclass(frozen=True)
class Completion:
    req_id: int
    hole_idx: int
    metres_drilled: float
