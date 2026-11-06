"""The interface the decision loop uses for the field side: limit check, approval, rig, lab."""

from __future__ import annotations

from typing import Protocol

from geoopt.protocol import CheckResult, Completion, Decision, RecommendationMsg


class Bridge(Protocol):
    async def next_req_id(self) -> int:
        """First request id this session may use."""
        ...

    async def submit(self, rec: RecommendationMsg) -> CheckResult:
        """Send a recommendation and get the limit check back."""
        ...

    async def await_approval(self, req_id: int) -> Decision:
        """Wait for the decision on a checked request."""
        ...

    async def await_completion(self, req_id: int) -> Completion:
        """Wait until the hole is drilled (its lab results are then released)."""
        ...
