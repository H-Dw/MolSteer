"""Compact expert inputs; full artifacts remain in host-owned state and tools."""
from copy import deepcopy
import json
from molsteer.common import digest


def task_context(event):
    result = {k: deepcopy(v) for k, v in (event or {}).items() if k != 'recovery'}
    recovery = (event or {}).get('recovery') or {}
    if recovery:
        result['recovery'] = {k: deepcopy(recovery[k]) for k in ('previous_attempt', 'previous_failure', 'candidate_review') if k in recovery}
        result['recovery']['artifact_access'] = 'read_expert_workspace(section="recovery") reads one canonical candidate; superseded direction copies are omitted.'
        result['recovery']['support_keys'] = list(recovery_support(event))
    return result


def recovery_candidate(event):
    recovery = (event or {}).get('recovery') or {}
    if recovery.get('full_model_authored_candidate'):
        return deepcopy(recovery['full_model_authored_candidate'])
    return deepcopy(recovery.get('mathematical_direction_candidates', {}))


def recovery_support(event):
    """Historical evidence and plans remain readable without automatic replay."""
    excluded = {'full_model_authored_candidate', 'mathematical_direction_candidates',
                'previous_attempt', 'previous_failure', 'candidate_review'}
    return {k: deepcopy(v) for k, v in ((event or {}).get('recovery') or {}).items() if k not in excluded}


def goal_index(biology):
    return {role: [d['direction_id'] for d in sorted(biology['directions'], key=lambda d: d['rank'])
                   if d['disposition'] == role] for role in ('optimize', 'constraint', 'deferred', 'monitor')}


def public_raw_analysis(analysis):
    # The host diagnostics/trajectory index already live behind raw-reference tools.
    return {k: deepcopy(v) for k, v in (analysis or {}).items() if k not in ('node_diagnostics', 'trajectory_index')}


def draft_index(drafts):
    return {key: dict(digest=digest(draft), status=draft.get('status'),
        formula_digest=digest(draft.get('formula')), observable_ids=[o.get('observable_id') for o in draft.get('observables', [])])
        for key, draft in drafts.items()}


def tool_receipt(event, *, include_feedback=True):
    """Retain actionable public feedback even when a large tool round is removed."""
    receipt = {k: event[k] for k in ('event_id', 'tool_name', 'summary') if k in event}
    value = event.get('output')
    if isinstance(value, dict):
        receipt['output_fields'] = list(value)
        receipt['collection_counts'] = {k: len(v) for k, v in value.items() if isinstance(v, (list, dict))}
        feedback = ('status', 'blocking', 'validated', 'passed')
        if include_feedback:
            feedback += ('hint', 'message', 'validation_rule', 'validation_path', 'validation_hint', 'validation_errors')
        receipt['feedback'] = {k: (value[k][:1600] if isinstance(value[k], str) else deepcopy(value[k]))
                               for k in feedback if k in value}
    return receipt


def compact_history(messages, memory, *, max_chars, recent_rounds):
    """Discard complete obsolete tool rounds; never orphan a tool response.

    Original recent AI messages remain unchanged (including provider reasoning
    signatures). Oversized rounds are replaced by receipts plus authoritative
    workspace state, with full public observations available by event ID.
    """
    from langchain_core.messages import HumanMessage
    prefix = messages[:2]
    history = messages[2:]
    starts = [i for i, m in enumerate(history) if getattr(m, 'type', None) == 'ai']
    groups = [history[a:b] for a, b in zip(starts, starts[1:]+[len(history)])]
    kept = groups[-recent_rounds:] if recent_rounds else []
    def size(groups):
        return sum(len(json.dumps(m.model_dump(), ensure_ascii=False, default=str)) for g in groups for m in g)
    while kept and size(kept) > max_chars:
        kept.pop(0)
    return [*prefix, HumanMessage(content=json.dumps({'current_workspace': memory}, ensure_ascii=False, allow_nan=False)),
            *(m for group in kept for m in group)]
