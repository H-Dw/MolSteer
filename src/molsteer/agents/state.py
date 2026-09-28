"""Typed, validated state exchanged by the MolSteer LangGraph workflow.

The state intentionally contains artifacts and decision summaries only. Provider
messages (including hidden reasoning blocks) never enter this object.
"""
from __future__ import annotations

import math
from typing import Any, Literal, TypedDict


class TraceEvent(TypedDict, total=False):
    event_id: str
    node: str
    kind: Literal["decision", "tool", "observation", "checkpoint", "error"]
    summary: str
    evidence_ids: list[str]
    tool_name: str
    input_digest: str
    output_digest: str
    timestamp: str


class MolSteerState(TypedDict, total=False):
    run_id: str
    packet: dict[str, Any]
    diagnostic_report: dict[str, Any]
    reward_spec: dict[str, Any]
    execution_request: dict[str, Any]
    execution_result: dict[str, Any]
    monitor_event: dict[str, Any]
    route: Literal["reader", "thinker", "executor", "monitor", "done", "error"]
    status: Literal["started", "planning", "executing", "monitoring", "completed", "failed"]
    trace: list[TraceEvent]
    errors: list[str]
    tool_results: dict[str, Any]
    step: int
    retunes: int
    replans: int
    checkpoint_path: str


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_packet(packet: Any) -> dict[str, Any]:
    """Validate a packet at the workflow boundary without mutating it.

    The domain validator is used when available; a small structural check remains
    available in lightweight/offline installations where RDKit is absent.
    """
    if not isinstance(packet, dict) or not isinstance(packet.get("packet_id"), str):
        raise ValueError("packet must be an object with packet_id")
    if not packet["packet_id"].strip():
        raise ValueError("packet_id cannot be blank")
    if "identity" in packet and not isinstance(packet["identity"], dict):
        raise ValueError("packet identity must be an object")
    if "observations" in packet and not isinstance(packet["observations"], list):
        raise ValueError("packet observations must be a list")
    if "representations" in packet and not isinstance(packet["representations"], dict):
        raise ValueError("packet representations must be an object")
    from molsteer.contracts import validate_enriched
    validate_enriched(packet)
    return packet


def validate_report(report: Any, packet: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("diagnostic report must be an object")
    if report.get("packet_id") not in (None, packet.get("packet_id")):
        raise ValueError("diagnostic report packet identity mismatch")
    for key in ("findings", "raw_state_findings", "evidence_gaps"):
        if key in report and not isinstance(report[key], list):
            raise ValueError(f"diagnostic report {key} must be a list")
    if "evidence_index" in report and not isinstance(report["evidence_index"], dict):
        raise ValueError("diagnostic report evidence_index must be an object")
    from molreader.localized_report import validate_localized_report
    validate_localized_report(report, packet)
    return report


def initial_state(*, run_id: str, packet: dict[str, Any] | None = None,
                  diagnostic_report: dict[str, Any] | None = None) -> MolSteerState:
    if not isinstance(run_id, str) or not run_id or len(run_id) > 160:
        raise ValueError("invalid run_id")
    packet = packet or {}
    if packet:
        validate_packet(packet)
    return {
        "run_id": run_id, "packet": packet,
        "diagnostic_report": diagnostic_report or {}, "reward_spec": {},
        "execution_request": {}, "execution_result": {}, "monitor_event": {},
        "route": "reader", "status": "started", "trace": [], "errors": [],
        "tool_results": {}, "step": 0, "retunes": 0, "replans": 0,
    }


__all__ = ["TraceEvent", "MolSteerState", "initial_state", "validate_packet", "validate_report"]
