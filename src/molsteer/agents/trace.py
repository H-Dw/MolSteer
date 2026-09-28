"""Redacted decision-trace and checkpoint persistence.

Trace payloads are deliberately summaries, digests and evidence IDs. Provider
messages, credentials, prompts containing secrets, and private reasoning are not
accepted for persistence.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from molsteer.common import digest

_ALLOWED_KINDS = {"decision", "tool", "observation", "checkpoint", "error"}
_SECRET_KEYS = re.compile(r"(?:api[_-]?key|secret|password|authorization|credential|private_reasoning|chain_of_thought|^thinking$|^reasoning$|^access_token$)", re.I)


def _redact(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): ("[REDACTED]" if _SECRET_KEYS.search(str(k)) else _redact(v)) for k, v in value.items()}
    if isinstance(value, list): return [_redact(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return {"unavailable": "nonfinite"}
    if isinstance(value, str): return value
    return value


def append_trace(state: dict[str, Any], *, node: str, kind: str, summary: str,
                 evidence_ids: list[str] | None = None, tool_name: str | None = None,
                 input_value: Any = None, output: Any = None) -> dict[str, Any]:
    if kind not in _ALLOWED_KINDS: raise ValueError(f"unsupported trace kind: {kind}")
    if not isinstance(summary, str) or not summary.strip(): raise ValueError("trace summary cannot be blank")
    trace = state.setdefault("trace", [])
    event = {"event_id": f"te_{digest({'run': state.get('run_id'), 'n': len(trace), 'summary': summary})[:24]}",
             "node": str(node)[:100], "kind": kind, "summary": summary[:1000],
             "evidence_ids": [str(x)[:200] for x in (evidence_ids or [])][:100],
             "timestamp": datetime.now(timezone.utc).isoformat()}
    if tool_name: event["tool_name"] = str(tool_name)[:160]
    if input_value is not None:
        event["input"] = _redact(input_value)
        event["input_digest"] = digest(event["input"])
    if output is not None:
        event["output"] = _redact(output)
        event["output_digest"] = digest(event["output"])
    trace.append(event)
    return state


def _payload(state: dict[str, Any]) -> dict[str, Any]:
    run_id = str(state.get("run_id", "run_unknown"))
    return {"schema_version": "1.1.0", "run_id": run_id, "status": state.get("status"),
            "route": state.get("route"), "step": int(state.get("step", 0)),
            "trace": _redact(state.get("trace", [])),
            "errors": [str(e)[:500] for e in state.get("errors", [])][:100],
            "packet_id": state.get("packet", {}).get("packet_id"),
            "reward_id": state.get("reward_spec", {}).get("reward_id"),
            "monitor_event": _redact(state.get("monitor_event", {})),
            "config_sha256": state.get("config_sha256"),
            "skill_sha256": state.get("skill_sha256"),
            "packet_sha256": digest(state.get("packet", {})),
            "report_sha256": digest(state.get("diagnostic_report", {}))}


def save_trace(state: dict[str, Any], trace_dir: str | Path) -> Path:
    run_id = str(state.get("run_id", "run_unknown"))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", run_id): raise ValueError("invalid run id")
    directory = Path(trace_dir); directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{run_id}.trace.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(_payload(state), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def save_checkpoint(state: dict[str, Any], trace_dir: str | Path) -> Path:
    run_id = str(state.get("run_id", "run_unknown"))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", run_id): raise ValueError("invalid run id")
    directory = Path(trace_dir); directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{run_id}.checkpoint.json"
    temporary = target.with_name(target.name + ".tmp")
    payload = _payload(state)
    payload["checkpoint_kind"] = "audit_artifact_snapshot"
    payload["automatic_resume_supported"] = False
    payload["artifacts"] = _redact({key: state.get(key) for key in (
        "packet", "diagnostic_report", "reward_spec", "plan", "validation",
        "execution_result", "config", "segments", "strength", "replans")})
    payload["artifacts_digest"] = digest(payload["artifacts"])
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle: value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str): raise ValueError("invalid checkpoint")
    if "artifacts" in value and digest(value["artifacts"]) != value.get("artifacts_digest"):
        raise ValueError("checkpoint artifact digest mismatch")
    return value


__all__ = ["append_trace", "save_trace", "save_checkpoint", "load_checkpoint"]
