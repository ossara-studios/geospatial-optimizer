"""The interface the decision loop uses to talk to the OT side (EAE or a stand-in)."""

from __future__ import annotations

from typing import Protocol

from geoopt.protocol import CheckResult, Completion, Decision, RecommendationMsg


class Bridge(Protocol):
    async def submit(self, rec: RecommendationMsg) -> CheckResult:
        """Send a recommendation and wait for EAE's limit check."""
        ...

    async def await_approval(self, req_id: int) -> Decision:
        """Wait for the operator's decision on a checked request."""
        ...

    async def await_completion(self, req_id: int) -> Completion:
        """Wait until the rig reports the dispatched hole complete (lab results released)."""
        ...
