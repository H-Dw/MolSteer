"""Current-state design assistance, separate from execution eligibility.

Experts own mechanisms, goals and mathematical choices. These helpers organize
public drafts and inspect their declared coverage; they never score future
outcomes, solve goal selection, or introduce a submission gate.
"""
from copy import deepcopy


BIOLOGY_GUIDE = {
    'starting_point': 'The current generation state, its contemporaneous prediction and MolReader evidence, plus an observed raw reference suffix only when configured. Guided outcomes remain unobserved.',
    'questions': [
        'Which measurements describe the same hypothesized mechanism, and which defects are independent?',
        'What does each goal directly repair, what independent defects does it cover, and what local feasibility can it improve?',
        'Which chemical hypotheses, editable variables and coupled risks qualify that potential repair benefit?',
        'Which smallest sufficient goal set covers the necessary mechanisms; what would remain uncovered if a goal were removed?',
        'Which relations must be preserved, and which current measurements could change the decision?',
        'Which current risks persist in sampled raw nodes; which clear later, recur or change chemical identity?',
        'Would earlier repair remove a necessary defect or disrupt useful native evolution; what current control channel can act on it?',
        'Which local changes accompany affinity/SA/stability trends, and what evidence is missing before assigning regional benefit?'],
    'stages': {
        'integration': 'Record mechanisms with evidence/finding IDs, competing explanations and couplings. Shared atom support is not a causal proof.',
        'candidate_goals': 'Optionally record goals as {goal_id, covers:[mechanism_id,...], repair_reason, risks, unknowns}; covers describes expert-declared mechanisms.',
        'selection': 'Optionally record selected_goal_ids and necessary_mechanism_ids, omitted alternatives and removal reasons. Coverage assistance is logical, not an efficacy forecast.',
        'raw_reference_review': 'Trace mechanisms through every configured raw node and current_condition_trajectory. Separate old-object comparability from later-molecule validity. Record references, persistence, native clearance/retyping, recurrence, useful regions and tradeoffs.',
        'value_ranking': 'Rank potential intervention value using persistence under current chemistry, severity, native repair timing, controllability, independent coverage, dependency and disruption of useful evolution. Explain pairwise priorities without invented intervention gains.',
        'revision': 'Respond to mathematical mechanism conflicts: accept, modify or reject proposed hypothesis/target/priority changes with reasons. Earlier biological assumptions are revisable.'},
    'handoff': 'Optimize directions are selected repairs; constraint directions are preservation. A scientifically necessary goal with missing execution inputs stays required=true, disposition=deferred, and proceeds to mathematics.',
    'scope': 'Editable evidence-linked decisions. The expert owns value ranking; raw counts are not utility scores. No fabricated outcome probabilities or additional eligibility gate.'}

MATH_GUIDE = {
    'workflow': ['Read the biologically selected goals and preservation conditions.',
        'Retrieve mechanisms and decide which source parts to retain, specialize or reconstruct.',
        'Determine mathematical target sets or justified optimization relations and their unresolved parameters.',
        'Derive local response: direction, active/stopping regions, curvature, symmetries and coupled effects.',
        'Construct candidates and compare their actual response before final compilation.'],
    'stages': {
        'source_transfer': 'Record inspected locators, native objects/roles, prerequisites, transferable parts and proposed changes.',
        'target_sets': 'Record observables, conditional chemical references, target relations and scale origins. A reference value does not determine a tolerance.',
        'local_response': 'Describe the desired derivative or constrained direction across relevant regimes before choosing a shape or aggregate.',
        'candidates': 'Record source-retained and newly derived parts, alternatives, local measurements and unresolved evaluator needs.'},
    'conditional_physics': {
        'construction': 'For each supported current chemical hypothesis q, construct normalized local geometry energy plus typed exclusion and justified native-contact preservation. Use only diagnosed coupled mechanisms.',
        'family_to_reason_about': 'Optional family: e_q=(E_geom,S^q-E_ref,S^q)/E_scale + lambda_s*sum(relu((d_min,ia^q-d_ia)/sigma_s)^2) + lambda_c*relu((C_native,S,t-C_S^q-delta_C)/sigma_C)^2. Retain only selected, evidenced mechanisms with evaluable observables; this is a reasoning example, not a preset objective or coefficient choice.',
        'geometry': 'Use current typed bond/angle references or a ready physical energy; bond_length_error and bond_angle_error rebind MMFF references at runtime. A local surrogate is not the entire MMFF energy.',
        'sterics': 'typed_steric_overlap uses current element radii and an explicit buffer; a current nearest distance is not an exclusion radius.',
        'contacts': 'Retain meaningful same-stage native contacts. native_nonincrease compares with the same-time unperturbed proposal on the CURRENT path, not the stored independent raw trajectory. A historical raw contact baseline needs an explicit evaluator; typed direction/area claims need actual observables.',
        'scales': 'Explain energy reference, normalization, buffers and allowed losses. Values are sourced or calibrated inputs, never invented from rank.',
        'hypotheses': 'Chemical transitions update interpretation. Do not freeze the initial graph or average incompatible reference bond lengths. Multiple-hypothesis mixtures need validated hypotheses and a suitable evaluator.',
        'collaboration': 'When physical references or local responses contradict biology, request_biology_revision with evidence and a concrete alternative before repairing the old formula.'},
    'scope': 'Public derivation workspace, not private reasoning. Partial records and any step order are allowed; no additional submission check.'}


def selected_directions(biology):
    """Scientific selections include necessary unresolved goals, not just runnable ones."""
    return [d for d in biology['directions'] if d['disposition'] in ('optimize', 'constraint')
            or (d['required'] and d['disposition'] == 'deferred')]


def goal_pools(biology):
    selected = selected_directions(biology)
    return {
        'scientific_goals': [deepcopy(d) for d in selected if d['disposition'] != 'constraint'],
        'preservation_goals': [deepcopy(d) for d in selected if d['disposition'] == 'constraint'],
        'researchable_goal_ids': [d['direction_id'] for d in biology['directions']],
        'compilation_candidate_ids': [d['direction_id'] for d in selected if d['disposition'] in ('optimize', 'constraint')],
        'unresolved_selected_goal_ids': [d['direction_id'] for d in selected if d['disposition'] == 'deferred'],
        'rule': 'Compilation candidates only have biological scope permission, not a validated mathematical evaluator. Research does not activate a direction. Necessary unresolved scientific goals remain in the design; execution needs present inputs and final tests, or an explicit biological revision.'}


def current_state_context(packet, report, dynamics, raw_reference_context=None, *, include_raw_reference=True):
    """Bind the problem without assigning causal regions or target values."""
    from .design_audit import bounded_values
    findings = []
    for origin in ('findings', 'raw_state_findings'):
        for finding in report.get(origin, []):
            row = bounded_values(finding, 4)
            row['report_group'] = origin
            findings.append(row)
    from molsteer.molreader.raw_reference import bound_raw_context, raw_reference_summary
    result = dict(identity=deepcopy(packet['identity']), representations=deepcopy(packet['representations']),
        model_dynamics=deepcopy(dynamics), findings=findings,
        observation_availability=[{k: deepcopy(m[k]) for k in ('metric_id', 'view', 'status')}
                                  for m in packet['observations']],
        chemical_readiness=deepcopy(packet['steering']['chemical_readiness']),
        interpretation='State and prediction describe this checkpoint in different representations. Prediction is not an observed terminal result. Separately configured raw suffix nodes are observations of the original path, not intervention outcomes. Reader grouping or raw co-evolution does not prove a shared cause or a regional benefit; missing measurements stay unavailable.')
    if include_raw_reference:
        result['raw_reference'] = raw_reference_summary(bound_raw_context(raw_reference_context, packet))
    return result


def new_workspace():
    return {'biology': {}, 'mathematics': {}}


def _string_ids(value):
    return list(dict.fromkeys(x for x in value if isinstance(x, str))) if isinstance(value, list) else []


def coverage_review(biology_workspace):
    """Inspect declared set coverage, not chemical sufficiency or intervention outcomes."""
    candidates = biology_workspace.get('candidate_goals', {}).get('goals', [])
    if not isinstance(candidates, list):
        candidates = []
    goals = {r['goal_id']: set(_string_ids(r.get('covers'))) for r in candidates
             if isinstance(r, dict) and isinstance(r.get('goal_id'), str)}
    selection = biology_workspace.get('selection', {})
    selected = _string_ids(selection.get('selected_goal_ids'))
    necessary_known = isinstance(selection.get('necessary_mechanism_ids'), list)
    necessary = set(_string_ids(selection.get('necessary_mechanism_ids')))
    known_coverage = {r['goal_id'] for r in candidates if isinstance(r, dict)
                      and isinstance(r.get('goal_id'), str) and isinstance(r.get('covers'), list)}
    unknown_coverage = [ident for ident in selected if ident not in known_coverage]
    complete = necessary_known and not unknown_coverage
    covered = set().union(*(goals.get(ident, set()) for ident in selected)) if selected else set()
    removals = []
    for ident in selected:
        remaining = set().union(*(goals.get(other, set()) for other in selected if other != ident))
        removals.append(dict(goal_id=ident,
            newly_uncovered_mechanism_ids=sorted((necessary & covered) - remaining) if complete else None,
            necessary_mechanism_ids_uncovered_after_removal=sorted(necessary - remaining) if complete else None))
    return dict(scope='expert_declared_mechanism_coverage_only',
        coverage_information='declared_sets_available' if complete else 'partial',
        necessary_mechanism_ids=sorted(necessary) if necessary_known else None,
        uncovered_mechanism_ids=sorted(necessary - covered) if complete else None,
        selected_goal_ids_without_coverage=unknown_coverage,
        selected_goal_ids_without_candidate_records=[ident for ident in selected if ident not in goals],
        removal_review=removals, selected_goal_ids=selected,
        limitation='Declared coverage does not establish independence, physical repair or global minimality. No goal is automatically selected, removed, ranked or weighted.')


def update_workspace(workspace, role, stage, record, direction_id=None):
    """Merge partial public records. Invalid advisory inputs never raise execution errors."""
    guide = BIOLOGY_GUIDE if role == 'biology' else MATH_GUIDE
    if stage not in guide['stages'] or not isinstance(record, dict) or not record:
        return dict(status='needs_input', blocking=False, stages=guide['stages'],
                    hint='Choose a listed stage and supply a nonempty partial public record; no stage is required for submission.')
    destination = workspace[role]
    if role == 'mathematics':
        if not isinstance(direction_id, str) or not direction_id:
            return dict(status='needs_input', blocking=False, hint='Supply a direction_id to keep derivations for different goals separate.')
        destination = destination.setdefault(direction_id, {})
    from .trace import _redact
    destination.setdefault(stage, {}).update(_redact(deepcopy(record)))
    result = dict(status='workspace', blocking=False, role=role, stage=stage,
                  record=deepcopy(destination[stage]), questions=guide['stages'])
    if role == 'biology':
        result['coverage_review'] = coverage_review(workspace['biology'])
    else:
        result['next_questions'] = [text for name, text in guide['stages'].items() if name not in destination]
    return result
