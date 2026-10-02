"""Host-bound temporal evidence tools shared by Reader and Thinker experts."""
from copy import deepcopy
from langchain_core.tools import tool
from molsteer.molreader.raw_reference import bound_raw_context, inspect_comparison


def raw_reference_tools(context, packet):
    frozen = deepcopy(bound_raw_context(context, packet))

    @tool
    def inspect_raw_comparison(section: str = 'summary', factor: str | None = None,
                               view: str | None = None, offset: int = 0, limit: int = 8) -> dict:
        """Page configured raw nodes, risk_tracks, regional_changes, outcome_trends, opportunities or factor_coverage; no inference runs."""
        return inspect_comparison(frozen, section, factor, view, offset, limit)

    @tool
    def read_raw_reference(reference_id: str, include_details: bool = False) -> dict:
        """Read exact saved cross-time evidence returned by inspect_raw_comparison; it is separate from current numerical evidence."""
        from molsteer.molreader.raw_reference import read_raw_reference as read
        return read(frozen, reference_id, include_details)

    @tool
    def inspect_raw_goal_trajectory(atom_ids: list[int] | None = None, evidence_ids: list[str] | None = None,
                                    factor: str | None = None, offset: int = 0, limit: int = 8) -> dict:
        """Trace a candidate mechanism through ALL configured raw nodes, including current typed references and terminal status."""
        from molsteer.molreader.raw_reference import inspect_goal_trajectory
        return inspect_goal_trajectory(frozen, atom_ids=atom_ids, evidence_ids=evidence_ids,
                                       factor=factor, offset=offset, limit=limit)

    return [inspect_raw_comparison, read_raw_reference, inspect_raw_goal_trajectory]
