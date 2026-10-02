"""MolReader agent: packet-bound diagnosis and independent metric-path tools."""
from __future__ import annotations
import json
from copy import deepcopy
from typing import Any, Callable
from langchain_core.tools import tool
from .state import validate_packet, validate_report
from .tools import make_statepack_tool
from .design_audit import bounded_values
from .raw_reference_tools import raw_reference_tools
from molsteer.molreader.raw_reference import bound_raw_context

class Reader:
    def __init__(self, *, model: Any = None, reader_fn: Callable[..., dict[str, Any]] | None = None, tools: list[Any] | None = None):
        if model is None and reader_fn is None: raise ValueError("Reader requires an API model or injected reader_fn")
        self.model, self.reader_fn, self.tools = model, reader_fn, tools or []
        self.bound_model = model.bind_tools(self.tools) if model is not None and self.tools and hasattr(model, "bind_tools") else model
    @staticmethod
    def _response(value: Any) -> dict[str, Any]:
        if isinstance(value, dict): return value
        content = getattr(value, "content", value)
        if isinstance(content, list): content = "".join(getattr(x, "text", str(x)) for x in content)
        if isinstance(content, str):
            try: parsed = json.loads(content)
            except json.JSONDecodeError: raise ValueError("Reader model must return a JSON diagnostic report") from None
            if isinstance(parsed, dict): return parsed
        raise ValueError("Reader response is not a diagnostic report object")
    def run(self, packet: dict[str, Any], *, report: dict[str, Any] | None = None) -> dict[str, Any]:
        validate_packet(packet)
        if report is None:
            if self.reader_fn is not None: report = self.reader_fn(packet)
            else:
                response = self.bound_model.invoke([{"role": "system", "content": "You are MolReader. Return only a validated diagnostic report JSON object with evidence IDs and explicit limitations; never propose edits."}, {"role": "user", "content": json.dumps({"packet_id": packet["packet_id"], "packet": packet}, sort_keys=True, default=str)}])
                report = self._response(response)
        validate_report(report, packet)
        return report

def reader_tools(packet: dict, supplied_report: dict | None = None, *, raw_reference_context: dict | None = None):
    validate_packet(packet); frozen = deepcopy(packet); result = {}; observed_paths = set()
    raw_context = bound_raw_context(raw_reference_context, frozen)
    result['raw_reference_context'] = deepcopy(raw_context)
    result['raw_reference_analysis'] = {
        'node_diagnostics': deepcopy(raw_context.get('node_diagnostics', [])),
        'trajectory_index': [{k: deepcopy(t[k]) for k in ('track_id', 'metric_id', 'view', 'atom_ids',
            'classification', 'current_condition_trajectory') if k in t} for t in raw_context.get('risk_tracks', [])],
        'interpretation': 'Host-read diagnostics at every configured node; expert interpretations can extend these observations.'}
    def metrics(path):
        observed_paths.add(path)
        groups = {"geometry": {"bond_lengths", "bond_angles", "mmff_local_geometry", "protein_clashes", "intramolecular_clashes"}, "chemistry": {"chemical_validity", "valence", "atom_inventory", "formal_charge", "connectivity", "chemistry_context", "structural_alerts", "mmff_strain", "posebusters"}}
        observations = frozen.get("observations", [])
        if path in groups: observations = [x for x in observations if x.get("metric_id") in groups[path]]
        else: observations = [x for x in observations if "confidence" in x.get("metric_id", "") or "entropy" in x.get("metric_id", "")]
        scoped = [dict(metric_id=m['metric_id'], view=m['view'], status=m['status'],
                       values=bounded_values(m['values']), thresholds=m.get('thresholds', {}),
                       evidence=deepcopy(m.get('evidence', [])), notes=m.get('notes', [])) for m in observations]
        return {"path": path, "observations": scoped, "availability": "observed" if observations else "unavailable", "limit": "Value arrays are explicitly bounded; evidence IDs and host packet are preserved. Use inspect_metric for a full metric. Missing metrics are not passes; no external programs launched"}
    @tool
    def inspect_metric(metric_id: str, view: str = 'prediction') -> dict:
        """Read one full bound metric when a bounded array or neighboring chemical context needs inspection."""
        matches = [m for m in frozen['observations'] if m['metric_id'] == metric_id and m['view'] == view]
        if len(matches) != 1:
            raise ValueError('Metric lookup must identify one bound observation')
        return deepcopy(matches[0])
    @tool
    def inspect_geometry() -> dict:
        """Read local geometry and pocket-clash evidence with original IDs."""
        return metrics("geometry")
    @tool
    def inspect_chemistry() -> dict:
        """Read chemical validity, strain and alert evidence without invention."""
        return metrics("chemistry")
    @tool
    def inspect_uncertainty() -> dict:
        """Read categorical confidence and entropy separately from defects."""
        data = metrics("uncertainty"); data["categorical_uncertainty"] = deepcopy(frozen.get("steering", {}).get("categorical_uncertainty", {})); return data
    @tool
    def record_raw_reference_analysis(record: dict) -> dict:
        """Keep optional public temporal interpretations, competing explanations and gaps; current diagnosis stays risk-only."""
        from .trace import _redact
        result.setdefault('raw_reference_analysis', {}).update(_redact(deepcopy(record)))
        return {'status': 'recorded', 'blocking': False,
                'analysis': deepcopy(result['raw_reference_analysis']),
                'limit': 'Temporal associations are descriptive; intervention selection belongs to MolThinker.'}
    @tool
    def submit_diagnosis() -> dict:
        """Build and validate DiagnosticReport after inspecting all metric paths."""
        if observed_paths != {"geometry", "chemistry", "uncertainty"}: raise ValueError("all three metric paths are required")
        if supplied_report is not None: report = deepcopy(supplied_report)
        else:
            from molreader.localized_report import make_localized_report
            report = make_localized_report(frozen)
        validate_report(report, frozen); result["report"] = report
        return {"status": "accepted", "packet_id": frozen["packet_id"], "report": report}
    return [make_statepack_tool(frozen), inspect_geometry, inspect_chemistry, inspect_uncertainty,
            inspect_metric, *raw_reference_tools(raw_context, frozen), record_raw_reference_analysis, submit_diagnosis], result

def reader_node(state: dict[str, Any], reader: Reader) -> dict[str, Any]:
    report = reader.run(state["packet"], report=state.get("diagnostic_report") or None)
    state["diagnostic_report"] = report; state["route"] = "thinker"; state["status"] = "planning"; state["step"] = state.get("step", 0) + 1
    from .trace import append_trace
    return append_trace(state, node="reader", kind="observation", summary="Validated packet diagnosis", evidence_ids=list(report.get("evidence_index", {})), output=report)

__all__ = ["Reader", "reader_node", "reader_tools"]
