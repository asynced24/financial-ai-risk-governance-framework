"""Shared gate result type used by every governance check.

Each governance module returns a :class:`GateResult` so the reporter can treat
fairness, drift, calibration, uncertainty and segment stability uniformly: one
verdict, the metrics behind it, and the threshold that produced it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import STATUS_ORDER, GateStatus


@dataclass
class GateResult:
    """The outcome of one governance gate for one model."""

    name: str
    status: GateStatus
    headline_metric: str
    headline_value: float
    threshold: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def symbol(self) -> str:
        return {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "headline_metric": self.headline_metric,
            "headline_value": self.headline_value,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "findings": self.findings,
            "details": self.details,
        }


def worst_status(statuses: Iterable[str]) -> GateStatus:
    """Return the most severe status in ``statuses`` (empty -> ``pass``)."""
    worst: GateStatus = "pass"
    for status in statuses:
        if STATUS_ORDER[status] > STATUS_ORDER[worst]:
            worst = status  # type: ignore[assignment]
    return worst


def aggregate_status(gates: dict[str, GateResult]) -> GateStatus:
    """Overall verdict across a model's gates: the worst individual verdict."""
    return worst_status(gate.status for gate in gates.values())
