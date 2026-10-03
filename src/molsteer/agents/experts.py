"""Sequential expert ReAct loops with bounded biological reconsideration."""
from copy import deepcopy
from typing import Any
from langchain_core.tools import tool
from molsteer.molthinker.expressions import OBSERVABLES, UNITS
from .expert_contracts import (BiologyPlan, MathematicalDesign, MathematicalDirection, validate_biology,
                               validate_math, compile_expert_spec)
from .researcher import ResearchService, RESEARCHER_INSTRUCTIONS
from .loop import run_tools
from .trace import append_trace
from .audit_contracts import FACTORS, DraftReplacement, ArchitectureAudit
from .design_audit import (biophysical_context, _evidence, inspect_bound_measurements, measurement_catalog,
                          replace_draft_fields, bounded_values, stage_direction_draft, assemble_direction_drafts,
                          retain_proposed_directions)
from molsteer.common import digest
from .decision_workspace import (BIOLOGY_GUIDE, MATH_GUIDE, current_state_context,
    new_workspace, update_workspace, selected_directions, goal_pools)
from .workflow_guidance import initial_role_reference, reference_catalog, read_reference
from .raw_reference_tools import raw_reference_tools
from molsteer.molreader.raw_reference import bound_raw_context, raw_reference_summary
from .reward_synthesis import (SYNTHESIS_GUIDE, function_card, construct_direction,
                               preview_architectures, derive_allocation_response)
from .expert_context import task_context, recovery_candidate, recovery_support, goal_index, public_raw_analysis, draft_index
from .priority import allocate_priorities
from .host_workspace import HostWorkspace, computational_content

def mathematical_draft_tool_schema(direction_only=False) -> dict:
    """Expose the full typed contract while allowing invalid drafts to be retained.

    A JSON-schema tool does not pre-validate its argument. The tool body applies
    MathematicalDesign and every scientific/numerical check before execution.
    Keeping definitions at the schema root also avoids Pydantic subset-model
    reconstruction losing recursive definitions through SkipValidation.
    """
    schema = (MathematicalDirection if direction_only else MathematicalDesign).model_json_schema()
    definitions = schema.pop('$defs', {})
    field = 'direction' if direction_only else 'design'
    return {'type':'object', 'properties':{field:schema}, 'required':[field],
            'additionalProperties':False, '$defs':definitions}

BIOLOGY_INSTRUCTIONS = """You are MolThinker's biology expert. Publish concise evidence-linked decisions, not private reasoning.
Start with the configured raw continuation's residual_needs index and counterevidence, then trace those needs to the current checkpoint.
For each candidate answer: What remains at the explicit raw final? Through which current coordinate mechanism could intervention help?
What independent necessary need would adding this goal cover? Inspect every configured node relevant to the mechanism, using current
chemistry, relation applicability, measured values, reference values and remaining deviations. Persistent, recurrent, late-emergent,
naturally resolved, relation-absent and unobserved conditions are different. Chemical identity change is an independent fact; it cannot
hide a defect under the new chemistry. Without an explicit final observation, terminal status is unknown.
Consider geometry, angles/torsions, strain, sterics, contact geometry and burial, identity, affinity and SA evidence where available.
Global co-evolution is correlation, not a local causal attribution or a measured intervention benefit. State competing explanations,
missing measurements and falsifiers. Use targeted existing measurements for decision-changing gaps. Future raw measurements do not
become current numeric evidence. Keep current DiagnosticReport semantics separate from temporal evidence.
Define the minimum sufficient goal set over remaining intervention needs. Compare independent needs before merging related measurements.
Shared atoms or an easy distance formula do not establish that one goal covers all needs. Deprioritize native-resolved defects; retaining
earlier intervention requires an explicit additional-value hypothesis and native-evolution interference analysis. Preserve beneficial
contacts and conformation changes. Zero, one or multiple optimization goals are legitimate; necessary unsupported goals stay deferred,
not replaced with an easy proxy. Record need IDs, precursor mechanism, independent coverage, counterevidence, useful evolution,
pairwise rank reasons and uncertainty in value_assessment and record_biology_decision. Rank is ordinal, not an invented gain probability.
Preservation that will influence guidance must be a selected optimize direction in the scalar reward, or a clause of one; independent
evaluation stays monitor. Legacy constraint directions are readable but require redesign, as there is no proposal acceptance controller.
Keep preservation_conditions empty for the scalar handoff; explain relationships in the decision records. Do not freeze initial chemistry.
When mathematics identifies a mechanism conflict, revisit assumptions, goals, preservation terms and ranks using its specific evidence.
Accept, modify or reject the suggested revision explicitly; retain unaffected goals. Use deferred for scientifically necessary missing
inputs, with clear reasons. Contracts are available by section on demand; partial decision records and tool read order are not gates.
Submit evidence-linked BiologyPlan with explicit finding dispositions. Do not infer future guided outcomes from raw continuation.
"""

MATH_INSTRUCTIONS = """You are MolThinker's mathematics expert. Publish derivations and evidence references, not private reasoning.
Deliver one differentiable scalar reward R for every remaining native FLOWR.ROOT step. Executor uses the same forward prediction and
self-conditioning, pulls R back to the current native coordinates, performs the native step and adds guidance_weight * dt * gradient(R).
The external weight is fixed for the run and applied once. Native active coordinates of the target are differentiable unless explicitly
fixed; observable atom support does not define the editable mask. Chemistry-conditioned references follow the declared current state.
First identify independent remaining intervention needs in biology_plan and their raw residual/precursor evidence. Ask whether a
native-resolved defect was selected without additional value and request a biology revision if needed. Do not let one convenient proxy
absorb independent needs. Future raw measurements are contextual and never current numeric evidence.
Research each mechanism in knowledge: source role, units, observables, physical reference and prerequisites; identify which parts
are retained, specialized or reconstructed. Keep original source formulas separate from the new construction.
Then determine target sets -> derive local response -> construct candidates. Sources do not set unexplained tolerances. A current distance
or future raw geometry is not a calibrated physical reference. Geometry terms can use current typed bond/angle references or supported
local energies; include torsion, exclusion or directional contacts only with suitable measurements and evaluators. Categorical changes,
SA and oracle scores have no invented coordinate derivative. Missing inputs remain design_only or trigger biology reconsideration.
Give independent losses physical scales before composition. Rank coefficients must enter the actual scalar expression through
priority_weights or justified scalar composition. Keep component values, scale provenance, coefficients, gradient norms, cosines and
actual directional contributions. Compare at the same current state, frame and allowed derivative space. Coordinate-copy response is
not live pullback; mixed independent views do not prove a shared live direction. Use the read-only live probe when available, otherwise
record not_run. Local surrogate improvement does not establish terminal quality or affinity benefit.
Construct preservation directly in R or explicitly leave it as independent evaluation. Use scalar_potential, not common_descent or
proposal constraints. Legacy common_descent and native_nonincrease designs require scalar redesign; no equivalent conversion is implied.
Executor never clips, normalizes, adapts weights, schedules budgets/windows, rejects proposals, backtracks, suspends on graph change or
revises R. Address excessive response through justified scale, curvature and composition design during candidate development. Compare
fixed external-weight response, cancellation and domination; do not rely on an executor controller. Natural zero gradients are valid
and reevaluated next step. An undefined required component is a calculation failure, never permission to silently drop it.
Use construct_direction_potential to save a candidate once. Read or patch its ID/version, compare response and test_staged_mathematical_design
when assembled; use test_current_candidate and submit_current_candidate for saved designs. Full legacy calls remain available through
select_tool_group(all), without a forced sequence. Reopen research for a concrete evidence gap while retaining the candidate. Contracts
have overview/section/full reads. Avoid repeated full schemas, stale drafts and resolved feedback. When mechanisms conflict with biology,
request_biology_revision with evidence and a concrete alternative. Explicitly defer missing scientific inputs instead of inventing them.
"""


def run_experts(runtime, state):
    materials = state.get('workflow_materials', {})
    instructions_by_role = {}
    for role, text in [('biology', BIOLOGY_INSTRUCTIONS), ('mathematics', MATH_INSTRUCTIONS)]:
        reference = initial_role_reference(materials, role)
        instructions_by_role[role] = text + ('\n\nGeneric background reference; current instructions on value rank, chemistry rebinding and revisable hypotheses supersede any conflicting older background advice:\n' + reference['text'] if reference else '')
    state['expert_guidance_sources'] = {role: {k: v for k, v in initial_role_reference(materials, role).items() if k in ('reference_id', 'sha256')}
                                      for role in instructions_by_role}
    state['expert_prompt_digests']={role:digest(text) for role,text in
        [*instructions_by_role.items(), ('researcher', RESEARCHER_INSTRUCTIONS)]}
    service = ResearchService(runtime, state)
    packet, report = state['packet'], state['diagnostic_report']
    measured = _evidence(packet, report)
    history = state.setdefault('expert_history', [])
    workspace_history = state.setdefault('expert_workspace_history', [])
    feedback = None
    raw_context = bound_raw_context(state.get('raw_reference_context'), packet)
    temporal_tools = raw_reference_tools(raw_context, packet)
    if getattr(runtime, 'measurement_supplements', None) is not None:
        temporal_tools.extend(runtime.measurement_supplements.tools())
    raw_summary = raw_reference_summary(raw_context)
    raw_analysis = public_raw_analysis(state.get('raw_reference_analysis')) if raw_context['status'] in ('available', 'partial') else {}
    current = current_state_context(packet, report, state['model_dynamics'], raw_context, include_raw_reference=False)
    carried_drafts, carried_biology = {}, {}
    staged_directions, proposed_draft, tested_draft, last_validation = {}, {}, {}, {}
    hosts = {role: HostWorkspace(state, 'thinker.'+role, runtime.config.runtime.no_progress_actions)
             for role in ('biology', 'mathematics')}
    numerical_cache = hosts['mathematics'].data.setdefault('numerical_cache', {})
    binding = {'packet':digest(packet), 'raw_comparison':raw_context.get('comparison_id')}
    for host in hosts.values():
        host.save('bound_evidence', binding, kind='evidence')
    recovered = recovery_candidate(state.get('monitor_event'))
    if recovered:
        hosts['mathematics'].save('recovery', recovered, kind='archive')

    def invoke(role, tools, context, completed):
        for function in tools:
            if function.name == 'get_expert_contract':
                function.invoke({})  # Deterministic contract version/overview initialization.
        run_tools(runtime._model('molthinker.'+role), tools, instructions=instructions_by_role[role],
                  context=context, state=state, node='thinker.'+role,
                  max_steps=runtime.config.runtime.max_agent_steps,
                  max_repairs=runtime.config.runtime.max_repairs, completed=completed,
                  working_memory=lambda: {'decisions':deepcopy(workspace['decisions'][role]),
                      'draft_index':draft_index(staged_directions) if role == 'mathematics' else {},
                       'latest_validation':bounded_values(last_validation),
                      'artifact_access':'read_expert_workspace reads current drafts, canonical recovery, decisions or archived public observations.'},
                  history_max_chars=runtime.config.thinker.history_max_chars,
                  history_recent_rounds=runtime.config.thinker.history_recent_rounds,
                  host_workspace=hosts[role])

    @tool
    def read_expert_workspace(section: str, direction_id: str | None = None,
                              event_id: str | None = None, path: list[str | int] | None = None) -> dict:
        """Read drafts/biology/decisions, canonical recovery, recovery_support or an archived observation; path selects a subtree. Historical recovery support is not current binding evidence."""
        sections = {'drafts':staged_directions, 'biology':state.get('biology_plan'),
                    'decisions':workspace['decisions'], 'recovery':recovery_candidate(state.get('monitor_event')),
                    'recovery_support':recovery_support(state.get('monitor_event'))}
        if section == 'observation':
            value = next((e.get('output') for e in state.get('trace', [])
                          if e.get('event_id') == event_id and e.get('kind') == 'tool'), None)
        else:
            value = sections.get(section)
        if direction_id and section == 'drafts':
            value = staged_directions.get(direction_id)
        elif section == 'drafts' and not path:
            value = draft_index(staged_directions)
        try:
            for key in path or []:
                value = value[key]
        except (KeyError, TypeError, IndexError):
            return {'status':'needs_input', 'blocking':False, 'hint':'Use an existing JSON path in the selected artifact.'}
        return {'status':'available' if value is not None else 'unavailable', 'section':section, 'value':deepcopy(value)}

    @tool
    def read_workflow_reference(reference_id: str, section: str | None = None) -> dict:
        """Read an existing configured workflow reference/section from the catalog; no arbitrary file access."""
        return read_reference(materials, reference_id, section)

    @tool
    def inspect_measurement(evidence_id: str, include_details: bool = False) -> dict:
        """Read one bound measurement; localized rows are exact, summary arrays are bounded by default."""
        return inspect_bound_measurements(measured,[evidence_id],include_details)['measurements'][0]

    @tool
    def inspect_measurements(evidence_ids: list[str], include_details: bool = False) -> dict:
        """Read 1-32 bound measurements in one call; full summary arrays require include_details=true."""
        return inspect_bound_measurements(measured,evidence_ids,include_details)

    @tool
    def list_measurement_references(metric_id: str, view: str = 'prediction', atom_id: int | None = None,
                                    offset: int = 0, limit: int = 12) -> dict:
        """Page reference IDs/support for a metric, optionally filtered by atom; inspect returned IDs for exact values."""
        return measurement_catalog(measured, metric_id, view, atom_id, offset, limit)

    for discussion in range(runtime.config.thinker.max_discussions+1):
        bio = {}
        inspected_factors = set()
        workspace = {'round': discussion, 'decisions': new_workspace(), 'source_transfers': [], 'constructions': [], 'architecture_previews': [],
                     'raw_reference_binding': {'comparison_id': raw_context.get('comparison_id'),
                                               'anchor_content_hash': raw_context['anchor_content_hash']}}
        workspace_history.append(workspace)

        @tool
        def get_expert_contract(section: str = 'overview', force_full: bool = False) -> dict:
            """Read contract overview or a named section; section=full explicitly reads the entire contract."""
            content = {'schema': BiologyPlan.model_json_schema(), 'factors': list(FACTORS),
                    'audit_required': runtime.config.thinker.require_design_audit}
            host = hosts['biology']
            receipt = host.save('contract', content, kind='contract')
            return dict(status='overview', sections=list(content), **receipt) if section == 'overview' and not force_full else host.read('contract', None if section in ('full', 'overview') else [section], force_full)

        @tool
        def inspect_biophysical_context(factor: str = 'all', view: str = 'both') -> dict:
            """Inspect current-state/prediction factors separately, with chemical readiness and unavailable inputs."""
            result = biophysical_context(packet, factor, view)
            inspected_factors.update(FACTORS if factor == 'all' else (factor,))
            return result

        @tool
        def record_biology_decision(stage: str, record: dict[str, Any]) -> dict:
            """Merge a partial public integration/candidate/selection record; inspect declared coverage without choosing goals."""
            return update_workspace(workspace['decisions'], 'biology', stage, record)

        @tool
        def submit_biology_plan(plan: BiologyPlan) -> dict:
            """Submit complete evidence-linked directions, ranks and diagnostic dispositions."""
            if bio:
                raise ValueError('Biology plan already submitted for this round')
            bio['plan'] = validate_biology(plan, report, packet,
                require_audit=runtime.config.thinker.require_design_audit)
            return {'status':'accepted', **hosts['biology'].save('biology', bio['plan'], kind='evidence')}

        invoke('biology', [get_expert_contract, inspect_biophysical_context, record_biology_decision, read_workflow_reference, read_expert_workspace, list_measurement_references, inspect_measurement, inspect_measurements, *temporal_tools, service.tool_for('biology'), submit_biology_plan],
               {'model_dynamics':state['model_dynamics'],
                'task_context':task_context(state.get('monitor_event')), 'revision_request':feedback,
                'raw_reference':raw_summary, 'current_state':current, 'decision_workspace':BIOLOGY_GUIDE, 'raw_reference_analysis':deepcopy(raw_analysis),
                'workflow_references':reference_catalog(materials),
                'previous_plan':state.get('biology_plan'), 'discussion_round':discussion}, lambda:bool(bio))
        state['biology_plan'] = bio['plan']
        biology = bio['plan']
        if feedback:
            old = carried_biology
            workspace['revision_resolution'] = {'requested':feedback,
                'changed_direction_ids':[d['direction_id'] for d in biology['directions'] if d != old.get(d['direction_id'])],
                'removed_direction_ids':sorted(set(old)-{d['direction_id'] for d in biology['directions']})}
        record = dict(round=discussion, biology_plan=deepcopy(biology))
        history.append(record)
        pools = goal_pools(biology)
        workspace['goal_pools'] = deepcopy(pools)
        if not selected_directions(biology):
            return {'deferral':{'status':'design_only', 'reason':'Biology selected no scientific repair or preservation goal'}}
        math_result = {}
        tested_designs = set()
        tested_draft = {}
        proposed_draft = {}
        staged_directions = {d['direction_id']:deepcopy(carried_drafts[d['direction_id']]) for d in biology['directions']
            if d['direction_id'] in carried_drafts and d == carried_biology.get(d['direction_id'])}
        hosts['mathematics'].retire_candidates()
        for ident, draft in staged_directions.items():
            hosts['mathematics'].save(ident, draft, {'packet':digest(packet), 'biology':digest(biology)})
        last_validation = {}
        targets = {d['direction_id'] for d in selected_directions(biology)}
        target_map = {d['direction_id']: d for d in biology['directions'] if d['direction_id'] in targets}
        research_map = {d['direction_id']: d for d in biology['directions']}

        @tool
        def record_function_derivation(direction_id: str, stage: str, record: dict[str, Any]) -> dict:
            """Retain partial source-transfer, target-set, local-response or candidate summaries without eligibility checks."""
            if direction_id not in research_map:
                return {'status':'needs_input', 'blocking':False, 'hint':'Use a direction_id from the biological plan.'}
            return update_workspace(workspace['decisions'], 'mathematics', stage, record, direction_id)

        @tool
        def get_expert_contract(section: str = 'overview', force_full: bool = False) -> dict:
            """Read versioned contract overview or named section; full and force_full are explicit opt-ins."""
            content = {'schema':MathematicalDesign.model_json_schema(), 'observables':OBSERVABLES,
                    'audit_required': runtime.config.thinker.require_design_audit,
                    'observable_fields':['observable_id','kind','view','atom_ids','evidence_ids','parameters'],
                    'construction':SYNTHESIS_GUIDE,
                    'parameters':{'receptor_distance':['receptor_serial','residue_id'],
                                  'typed_steric_overlap':['receptor_serial','residue_id','buffer_ratio'],
                                  'bond_length_error':[], 'bond_angle_error':[],
                                  'anchor_offset':['reference (three-vector)','origin'],
                                  'direction_alignment':['reference (three-vector)','origin'],
                                  'others':[]},
                    'expression':{'leaf':{'op':'observable','id':'observable_id'},
                                  'constant':{'op':'constant','value':1.0,'unit':'angstrom','origin':'evidence or declared scale origin'},
                                  'operator':{'op':'subtract','args':['expression','expression']},
                                  'unary':['relu','abs','sqrt','sin','cos','power'],
                                  'binary':['subtract','divide','periodic_difference'],
                                  'associative':['add','multiply','maximum','minimum'],
                                  'reduce':['sum','mean'], 'power_extra_field':'exponent in [0.5,8]',
                                  'units':list(UNITS), 'limits':'128 nodes, depth 12; scalar nonnegative dimensionless output'},
                    'strategy_examples':[{'mode':'scalar_potential','aggregation':{'op':'single'},'justification':'Case-specific reason required'},
                                         {'mode':'scalar_potential','aggregation':{'op':'weighted_sum'},'priority_weights':{'goal_a':1.0,'goal_b':0.5},'justification':'Explain value ranking and normalized marginal response'},
                                         {'mode':'design_only','aggregation':None,'justification':'Retain unresolved scalar-function needs without execution'}],
                    'state_context':{'representations':bounded_values(packet['representations']),
                                     'chemical_readiness':packet['steering']['chemical_readiness'],
                                     'receptor_atom_count':len(packet['steering']['receptor_atoms']),
                                     'support_rule':'Use exact localized measurement references for atom/receptor bindings; full receptor coordinates are retained by the Executor, not repeated in this contract.',
                                     'graph_signatures':packet['steering']['graph_signatures']}}
            host = hosts['mathematics']
            receipt = host.save('contract', content, kind='contract')
            return dict(status='overview', sections=list(content), **receipt) if section == 'overview' and not force_full else host.read('contract', None if section in ('full', 'overview') else [section], force_full)

        @tool
        def get_function_catalog() -> dict:
            """List all reviewed knowledge families and prerequisites before choosing and retrieving formulas."""
            return {'functions': [{k: c.get(k) for k in ('function_id', 'name_en', 'role', 'tags', 'prerequisites')}
                    for c in service.local.chunks.values() if c.get('function_id')],
                    'rule': 'Catalog entries are not inspected formulas. Retrieve selected and relevant alternative rows with search_direction_knowledge.'}

        @tool
        def prepare_function_synthesis(direction_id: str, function_ids: list[str] | None = None,
                                       query: str | None = None) -> dict:
            """Retrieve by mechanism query or chosen source IDs before determining targets and local response."""
            known = {row.get('function_id') for row in service.local.chunks.values() if row.get('function_id')}
            if (direction_id not in research_map or function_ids is not None and
                    (not 1 <= len(function_ids) <= 8 or len(set(function_ids)) != len(function_ids) or not set(function_ids) <= known)):
                return {'status': 'needs_input', 'blocking': False,
                    'hint': 'Use a biological direction_id and a mechanism query, or 1-8 distinct source IDs from the catalog. Deferred goals can be researched.'}
            question = query.strip() if isinstance(query, str) and query.strip() else (
                research_map[direction_id]['mechanism']+' '+research_map[direction_id]['optimization_direction'])[:2000]
            retrieved = ([service.local_search(direction_id, question, ident) for ident in function_ids]
                         if function_ids is not None else [service.local_search(direction_id, question)])
            result = {'status': 'workspace', 'direction_id':direction_id, 'retrievals': [{
                'retrieval_id': r['retrieval_id'], 'function_cards': [function_card(row) for row in r['records'] if row.get('function_id')]} for r in retrieved],
                'source_passages':[deepcopy(row) for r in retrieved for row in r['records'] if not row.get('function_id')],
                'next_questions': {name: MATH_GUIDE['stages'][name] for name in ('source_transfer','target_sets','local_response')},
                'scope':'Actual sources and optional derivation help. Source availability does not activate a biological goal or supply its target parameters.'}
            workspace['source_transfers'].append(deepcopy(result))
            return result

        @tool
        def construct_direction_potential(direction_id: str, relations: list[dict[str, Any]],
                                           within_direction: str, interpretation: dict[str, Any] | None = None) -> dict:
            """Construct a saved draft. Each relation has function_id, shape, clause_ids,
            observable={observable_id,kind,view,atom_ids,evidence_ids,parameters}, and
            parameters={shape parameter name: parameter record}. Read construction
            contract and inspected function serialization_options for exact names.
            current_reference_quadratic only takes scale in the observable unit;
            bond_length_error/bond_angle_error observable parameters are {}.
            Do not use name/units/observable_kind as observable field aliases.
            """
            known = {row.get('function_id') for row in service.local.chunks.values() if row.get('function_id')}
            if direction_id not in targets or not 1 <= len(relations) <= 5:
                return {'status': 'needs_input', 'blocking': False, 'hint': 'Use an active direction_id and 1-5 relations; custom larger expressions can use the existing draft tools.'}
            ids = [r.get('function_id') for r in relations if isinstance(r, dict)]
            if len(ids) != len(relations) or any(not isinstance(ident, str) or ident not in known for ident in ids):
                return {'status': 'needs_input', 'blocking': False, 'hint': 'Select actual function IDs; consult the catalog/transfer cards before choosing a mechanism.'}
            ids = list(dict.fromkeys(ids))
            retrieved = []
            for ident in ids:
                existing = next((r for r in reversed(state['retrieval_records']) if r['direction_id'] == direction_id and
                    any(row.get('function_id') == ident for row in r['records'])), None)
                retrieved.append(existing or service.local_search(direction_id, 'Construct selected relation '+ident, ident))
            result = construct_direction(target_map[direction_id], relations, within_direction,
                retrieved, packet, interpretation)
            if result['status'] == 'draft_only':
                if target_map[direction_id]['disposition'] == 'deferred':
                    result['direction'].update(status='design_only', missing_requirements=list(dict.fromkeys(
                        [need for factor in biology.get('factor_assessment', []) if direction_id in factor['direction_ids']
                         for need in factor['missing_requirements']]
                        or ['Selected biology goal retains deferred execution scope; resolve its inputs and request explicit biology revision before activation.'])))
                staged_directions[direction_id] = deepcopy(result['direction'])
                tested_draft.clear(); last_validation.clear()
            result['derivation_record'] = deepcopy(workspace['decisions']['mathematics'].get(direction_id, {}))
            if result['status'] == 'draft_only':
                receipt = hosts['mathematics'].save(direction_id, staged_directions[direction_id], {'packet':digest(packet), 'biology':digest(biology)})
                workspace['constructions'].append(receipt)
                return dict(status='draft_only', **receipt, next_action='Read or patch by candidate ID; construction already saved it.')
            return result

        @tool
        def compare_constructed_architectures(strategies: list[dict[str, Any]]) -> dict:
            """Measure alternative scalar compositions on coordinate copies and available live adapters; report component responses and conflicts."""
            strategies = [allocate_priorities({'directions':list(staged_directions.values()),'strategy':s}, biology,
                            runtime.config.thinker.rank_decay)['strategy'] for s in strategies]
            result = preview_architectures(packet, biology, staged_directions, state['model_dynamics'], strategies)
            from molsteer.molexecutor.expert_control import probe_live_scalar_response
            adapter = getattr(runtime.inference_adapter, 'sampling_adapter', runtime.inference_adapter)
            result['live_response'] = [probe_live_scalar_response(adapter, packet, biology, staged_directions, s) for s in strategies]
            workspace['architecture_previews'].append(deepcopy(result))
            return result

        @tool
        def derive_allocation_operator(first_deficit: float, second_deficit: float,
                                       desired_pressure_ratio: float, reason: str) -> dict:
            """Derive a norm exponent from an expert-chosen marginal response law; never choose weights or clamp unsupported designs."""
            result = derive_allocation_response(first_deficit, second_deficit, desired_pressure_ratio, reason)
            workspace.setdefault('allocation_derivations', []).append(deepcopy(result))
            return result

        @tool
        def search_direction_knowledge(direction_id: str, query: str, function_id: str | None = None) -> dict:
            """Retrieve original local formulas for any biological direction, including unimplemented scientific goals."""
            if direction_id not in research_map:
                raise ValueError('Unknown selected biological direction')
            return service.local_search(direction_id, query, function_id)

        @tool
        def inspect_direction_functions(direction_id: str, function_ids: list[str]) -> dict:
            """Inspect 1-12 catalog functions for a biological direction; research does not activate it."""
            if direction_id not in research_map:
                raise ValueError('Unknown selected biological direction')
            known={row.get('function_id') for row in service.local.chunks.values() if row.get('function_id')}
            if not 1<=len(function_ids)<=12 or len(set(function_ids))!=len(function_ids) or not set(function_ids)<=known:
                raise ValueError('Use 1-12 distinct function IDs returned by the catalog')
            return {'retrievals':[service.local_search(direction_id,'Inspect candidate '+ident,ident) for ident in function_ids]}

        @tool
        def request_biology_revision(direction_ids: list[str], issue: str, requested_change: str,
                                     evidence_ids: list[str] | None = None, proposed_changes: dict[str, Any] | None = None) -> dict:
            """Challenge earlier biology with a mechanism conflict, evidence and proposed hypothesis/target/rank changes; biology reviews the proposal next."""
            if math_result or not direction_ids or not set(direction_ids) <= set(research_map) or min(len(issue),len(requested_change)) < 12:
                raise ValueError('Revision must identify selected directions and a concrete issue/change')
            math_result['revision'] = dict(direction_ids=direction_ids, issue=issue, requested_change=requested_change,
                evidence_ids=evidence_ids or [], proposed_changes=deepcopy(proposed_changes or {}),
                mathematical_evidence={ident:deepcopy(workspace['decisions']['mathematics'].get(ident, {})) for ident in direction_ids},
                response_requested='Reconsider mechanism, target, preservation and rank; explain accepted, modified or rejected changes. Earlier plans are revisable.')
            return {'status':'revision_requested', **math_result['revision']}

        @tool(args_schema=mathematical_draft_tool_schema())
        def test_mathematical_design(design: dict[str, Any]) -> dict:
            """Check a proposed scalar handoff and test its expressions and component responses on coordinate copies."""
            if not isinstance(design,(dict,MathematicalDesign)):
                raise ValueError('Propose a complete mathematical draft before patching')
            proposed_draft['design']=deepcopy(design.model_dump() if isinstance(design,MathematicalDesign) else design)
            hosts['mathematics'].save('design', proposed_draft['design'], {'packet':digest(packet), 'biology':digest(biology)})
            tested_draft.clear(); last_validation.clear()
            retain_proposed_directions(staged_directions,proposed_draft['design'],targets)
            checked=validate_math(design,biology,packet,state['retrieval_records'],service.sources(), report=report,
                require_audit=runtime.config.thinker.require_design_audit, rank_decay=runtime.config.thinker.rank_decay)
            spec,deferred=compile_expert_spec(packet,report,biology,checked,state['retrieval_records'],state['model_dynamics'],service.sources())
            if deferred:
                tested_draft['design']=deepcopy(checked)
                return deferred
            from .executor import validate_and_test_reward
            biology_dependencies=[{k:d.get(k) for k in ('direction_id','rank','disposition','evidence_ids','preservation_conditions')}
                                  for d in biology['directions']]
            cache_key=digest((computational_content(checked), packet, biology_dependencies, state['model_dynamics']))
            validation=deepcopy(numerical_cache.get(cache_key))
            if validation is None:
                validation=validate_and_test_reward(packet,spec,report)
                numerical_cache[cache_key]=deepcopy(validation)
            else:
                validation['reward_id']=spec['reward_id']
            last_validation.clear(); last_validation.update(deepcopy(validation))
            if validation.get('passed'):
                tested_designs.add(digest(checked))
                tested_draft['design']=deepcopy(checked)
            return validation

        @tool
        def patch_and_test_mathematical_design(changes: list[DraftReplacement]) -> dict:
            """Replace 1-16 existing JSON fields of the last proposed draft, then rerun every contract and numerical test."""
            if not proposed_draft:
                raise ValueError('Propose a complete mathematical draft before patching')
            candidate=replace_draft_fields(proposed_draft['design'],
                [c.model_dump() if isinstance(c,DraftReplacement) else c for c in changes])
            return test_mathematical_design.invoke({'design':candidate})

        @tool(args_schema=mathematical_draft_tool_schema(direction_only=True))
        def stage_mathematical_direction(direction: dict[str, Any]) -> dict:
            """Retain one complete selected-direction draft only; never authorize execution or claim validation."""
            result = stage_direction_draft(staged_directions, direction, targets)
            ident = direction['direction_id']
            receipt = hosts['mathematics'].save(ident, staged_directions[ident], {'packet':digest(packet), 'biology':digest(biology)})
            tested_draft.clear(); last_validation.clear()
            return {'status':'draft_only', **receipt}

        @tool
        def test_staged_mathematical_design(strategy: dict[str, Any], conflict_assessment: str,
                                             independent_evaluation: str, design_audit: ArchitectureAudit) -> dict:
            """Assemble all model-authored direction drafts and run every full-design scientific and numerical check."""
            candidate=assemble_direction_drafts(staged_directions, biology, dict(strategy=strategy,
                conflict_assessment=conflict_assessment, independent_evaluation=independent_evaluation,
                design_audit=design_audit.model_dump() if isinstance(design_audit,ArchitectureAudit) else design_audit))
            return test_mathematical_design.invoke({'design':candidate})

        @tool
        def submit_mathematical_design(design: MathematicalDesign | None = None) -> dict:
            """Submit the last passing or explicitly deferred artifact; deferred submission never authorizes execution."""
            if math_result:
                raise ValueError('Mathematics already submitted for this round')
            if design is None:
                if not tested_draft:
                    raise ValueError('test_mathematical_design must pass for the exact submitted artifact')
                design=tested_draft['design']
            checked=validate_math(design, biology, packet, state['retrieval_records'], service.sources(), report=report,
                require_audit=runtime.config.thinker.require_design_audit, rank_decay=runtime.config.thinker.rank_decay)
            spec,deferred=compile_expert_spec(packet,report,biology,checked,state['retrieval_records'],state['model_dynamics'],service.sources())
            if not deferred and digest(checked) not in tested_designs:
                raise ValueError('test_mathematical_design must pass for the exact submitted artifact')
            math_result['design'] = checked
            return {'status':'accepted', **hosts['mathematics'].save('design', checked, {'packet':digest(packet), 'biology':digest(biology)})}

        @tool
        def read_candidate(candidate_id: str, path: list[str | int] | None = None, force_full: bool = False) -> dict:
            """Read a canonical candidate by ID, optionally selecting exact fields."""
            return hosts['mathematics'].read(candidate_id, path, force_full)

        @tool
        def patch_candidate(candidate_id: str, changes: list[DraftReplacement], version: int | None = None) -> dict:
            """Patch existing candidate fields without resending the candidate. Test explicitly when ready."""
            host = hosts['mathematics']
            saved = host.data['artifacts'].get(candidate_id)
            if saved is None or version is not None and version != saved['version']:
                return dict(status='needs_input', blocking=False, hint='Read the current candidate ID/version.')
            source = host.read(candidate_id, force_full=True)['value']
            value = replace_draft_fields(source, [c.model_dump() for c in changes])
            if candidate_id == 'design':
                proposed_draft['design'] = value
                retain_proposed_directions(staged_directions, value, targets)
            else:
                staged_directions[candidate_id] = value
                if proposed_draft:
                    for i, direction in enumerate(proposed_draft['design']['directions']):
                        if direction['direction_id'] == candidate_id:
                            proposed_draft['design']['directions'][i] = deepcopy(value)
            tested_draft.clear(); last_validation.clear()
            return dict(status='draft_only', **host.save(candidate_id, value, saved['dependencies']))

        @tool
        def test_current_candidate(candidate_id: str = 'design', version: int | None = None) -> dict:
            """Test a saved complete design by reference; individual directions are assembled with test_staged_mathematical_design."""
            saved = hosts['mathematics'].data['artifacts'].get(candidate_id)
            if not saved or candidate_id != 'design' or version is not None and saved['version'] != version:
                return dict(status='needs_input', blocking=False, hint='Assemble directions or select the current complete design version.')
            return test_mathematical_design.invoke({'design': saved['content']})

        @tool
        def submit_current_candidate() -> dict:
            """Submit the exact last tested or explicitly deferred design without retransmitting its schema."""
            return submit_mathematical_design.invoke({})

        math_tools = [get_expert_contract, get_function_catalog, prepare_function_synthesis, record_function_derivation, read_workflow_reference, read_expert_workspace, *temporal_tools, construct_direction_potential,
                               compare_constructed_architectures, derive_allocation_operator, inspect_biophysical_context, list_measurement_references, inspect_measurement, inspect_measurements,
                               search_direction_knowledge, inspect_direction_functions, service.tool_for('mathematics'),
                               request_biology_revision, stage_mathematical_direction, test_staged_mathematical_design,
                               patch_and_test_mathematical_design, submit_mathematical_design,
                               read_candidate, patch_candidate, test_current_candidate, submit_current_candidate, test_mathematical_design]
        invoke('mathematics', math_tools,
               {'biology_plan':biology, 'model_dynamics':state['model_dynamics'],
                'task_context':task_context(state.get('monitor_event')),
                'current_state':current, 'goal_index':goal_index(biology),
                'rank_allocation':{'default_decay':runtime.config.thinker.rank_decay,
                    'meaning':'Ordinal preference scaling; provide explicit priority_weights to override after response analysis.'},
                'raw_reference':raw_summary, 'raw_reference_analysis':deepcopy(raw_analysis),
                'synthesis_workspace':SYNTHESIS_GUIDE,
                'function_derivation':MATH_GUIDE, 'biology_decisions':deepcopy(workspace['decisions']['biology']),
                'workflow_references':reference_catalog(materials),
                'research_packets':state['research_packets'], 'remaining_discussions':runtime.config.thinker.max_discussions-discussion},
               lambda:bool(math_result))
        record.update(deepcopy(math_result), retrievals=deepcopy(state['retrieval_records']),
                      research_packets=deepcopy(state['research_packets']), goal_pools=deepcopy(pools))
        if 'revision' in math_result:
            feedback = math_result['revision']
            carried_drafts = deepcopy(staged_directions)
            carried_biology = {d['direction_id']:deepcopy(d) for d in biology['directions']}
            if discussion == runtime.config.thinker.max_discussions:
                return {'deferral':{'status':'design_only', 'reason':'Expert discussion budget exhausted', 'revision':feedback}}
            continue
        state['mathematical_design'] = math_result['design']
        spec, deferral = compile_expert_spec(packet, report, biology, math_result['design'],
                                             state['retrieval_records'], state['model_dynamics'], service.sources())
        append_trace(state, node='thinker', kind='decision', summary='Biology priorities and mathematical design bound to evidence',
                     output={'biology_plan':biology, 'mathematical_design':math_result['design'], 'deferral':deferral})
        return {'deferral':deferral} if deferral else {'spec':spec, 'validation':deepcopy(last_validation)}
    raise RuntimeError('Unreachable expert scheduling state')
