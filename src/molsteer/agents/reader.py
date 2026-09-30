"""MolReader agent: packet-bound diagnosis and independent metric-path tools."""
from __future__ import annotations
import json
from copy import deepcopy
from typing import Any, Callable
from langchain_core.tools import tool
from .state import validate_packet, validate_report
from .tools import make_statepack_tool

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

def reader_tools(packet: dict, supplied_report: dict | None = None):
    validate_packet(packet); frozen = deepcopy(packet); result = {}; observed_paths = set()
    def metrics(path):
        observed_paths.add(path)
        groups = {"geometry": {"bond_lengths", "bond_angles", "mmff_local_geometry", "protein_clashes", "intramolecular_clashes"}, "chemistry": {"chemical_validity", "valence", "atom_inventory", "formal_charge", "connectivity", "chemistry_context", "structural_alerts", "mmff_strain", "posebusters"}}
        observations = frozen.get("observations", [])
        if path in groups: observations = [x for x in observations if x.get("metric_id") in groups[path]]
        else: observations = [x for x in observations if "confidence" in x.get("metric_id", "") or "entropy" in x.get("metric_id", "")]
        return {"path": path, "observations": deepcopy(observations), "availability": "observed" if observations else "unavailable", "limit": "Missing metrics are not passes; no external programs launched"}
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
    def submit_diagnosis() -> dict:
        """Build and validate DiagnosticReport after inspecting all metric paths."""
        if observed_paths != {"geometry", "chemistry", "uncertainty"}: raise ValueError("all three metric paths are required")
        if supplied_report is not None: report = deepcopy(supplied_report)
        else:
            from molreader.localized_report import make_localized_report
            report = make_localized_report(frozen)
        validate_report(report, frozen); result["report"] = report
        return {"status": "accepted", "packet_id": frozen["packet_id"], "report": report}
    return [make_statepack_tool(frozen), inspect_geometry, inspect_chemistry, inspect_uncertainty, submit_diagnosis], result

def reader_node(state: dict[str, Any], reader: Reader) -> dict[str, Any]:
    report = reader.run(state["packet"], report=state.get("diagnostic_report") or None)
    state["diagnostic_report"] = report; state["route"] = "thinker"; state["status"] = "planning"; state["step"] = state.get("step", 0) + 1
    from .trace import append_trace
    return append_trace(state, node="reader", kind="observation", summary="Validated packet diagnosis", evidence_ids=list(report.get("evidence_index", {})), output=report)

__all__ = ["Reader", "reader_node", "reader_tools"]
