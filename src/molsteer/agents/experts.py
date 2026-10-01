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
                          replace_draft_fields, bounded_values, stage_direction_draft, assemble_direction_drafts)
from molsteer.common import digest
from .decision_workspace import (BIOLOGY_GUIDE, MATH_GUIDE, current_state_context,
    new_workspace, update_workspace, selected_directions, goal_pools)
from .workflow_guidance import initial_role_reference, reference_catalog, read_reference
from .raw_reference_tools import raw_reference_tools
from molsteer.molreader.raw_reference import bound_raw_context, raw_reference_summary
from .reward_synthesis import (SYNTHESIS_GUIDE, function_card, construct_direction,
                               preview_architectures, derive_allocation_response)

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

BIOLOGY_INSTRUCTIONS = '''You are MolThinker's biology expert. Select the smallest sufficient
scientific repair-goal set from the current generator checkpoint and MolReader diagnosis. Produce
public evidence-linked decisions, not private thought traces. State, contemporaneous prediction
and an observed final result are different objects. No paired intervention outcomes are supplied;
do not invent terminal gains or probabilities of persistence, success or retention.

When raw_reference is available or partial, a configured suffix of the SAME raw path has already
been observed. Read its comparison overview and inspect_raw_comparison/read_raw_reference when
details matter. Compare persistent defects, sampled clearance intervals, recurrence and later risks;
compare contact/burial, conformational and chemical evolution with configured affinity, SA and
stability trends. Integrate these candidates with the current diagnosis, not as a separate formula
list. For raw persistent risks ask what native continuation failed to resolve. For later clearance
ask whether earlier repair is locally controllable and would conflict with useful native rearrangement.
For region enhancement require a plausible mechanism and present control channel; a global score
change alone cannot assign causal benefit to that region. Changed elements, charge or topology are
chemical transitions, not successful movement of the old chemical object. Coordinate gradients
cannot promise categorical groups or improved synthesis feasibility. These raw outcomes are valid
reference observations, while guided outcomes and the benefit of acting earlier remain unobserved.
No risk/score ranking or intervention is selected by the comparison helper. Keep unknowns and
counter-explanations and choose the smallest sufficient current goal set, including preservation.
Optionally record raw_reference_review with comparison_id/locators and selection/omission reasons;
this editable public decision is passed to mathematics and never becomes an additional gate.

Integrate the problem before selecting a formula:
1. Read current_state and inspect_biophysical_context. Keep representations and chemical hypotheses
separate. Assess bond geometry, angle/torsion/stereochemistry, intramolecular stability, sterics,
target contacts, surface/burial, identity/functional groups and task relevance. Consolidate related
observations into mechanisms with competing explanations. Shared atoms or one Reader group alone
do not prove a common cause; a stretched bond does not explain all conformational strain.
2. Consider independent candidate goals by potential repair benefit supported by current evidence:
direct repair, independent defect coverage, local feasibility, controllability and coupled risks.
Distinguish observations, mechanistic hypotheses and missing facts. Qualitative/conditional priorities
are valid. Severity, alert count and implementation convenience do not determine scientific value.
3. Select sufficient coverage without redundant goals. Ask what necessary mechanism remains
unhandled after removing each goal, and whether a proxy could improve while its defect persists.
Keep necessary preservation and decision-changing unknowns separate. One goal can suffice; several
independent defects may need several. Do not freeze the whole intermediate pose or chemical slots.
4. Use record_biology_decision for editable integration, candidate and selection summaries when
helpful. Its coverage review only inspects your declared mechanism sets; it does not prove chemistry,
choose goals or rank benefits. Partial records and omitted helper calls do not block submission.
5. Submit BiologyPlan using summary/priority_reason for the integration, repair rationale, removal
reasons and uncertainty. Optimize means selected repair; constraint means necessary preservation.
A selected scientific goal lacking current execution inputs remains required=true, disposition=deferred
with its actual missing facts. Mathematics can research it. Do not discard it to fit a distance backend.
rank is a stable handoff order, not a value score, reward coefficient or activation schedule.

The existing schema is available on demand. Record all finding dispositions and factor assessments.
Use exact localized measurement IDs, including finding_ids=[] for independently measured context.
repair_clauses cover the declared predicates; their observable kinds may name an unsupported design
need. State the biological relation first; mathematics determines evidenced numerical target sets.
Screening cutoffs are not calibrated tolerances. Coordinate control alone cannot induce a functional
group or verify an entire contact network from two pairs. Use request_research for a decision-changing
current evidence gap. Unrun live or terminal evaluation remains unknown. On revision, change the
smallest necessary goal/scope using the returned mathematical evidence; on graph change reconsider
chemical roles and references under the original task. Generic background references are design
material, not additional submission gates or prior outcome information.'''

MATH_INSTRUCTIONS = '''You are MolThinker's mathematics expert. Construct case-specific functions
from the biologically selected goals, current-state mechanisms, preservation and uncertainty.
Produce public derivation summaries, not private thought traces. Work forward:

Configured raw_reference describes an observed original suffix, not a guided counterfactual.
Read biological raw_reference_review and the selected goal's temporal support when available;
use the shared temporal tools to inspect mechanisms before source transfer. A raw contact/fragment
or screening improvement may motivate a hypothesis, but raw final coordinates, distances and graph
are not automatically acceptable sets, scales or numerical targets. Construct the function on
CURRENT bound measurements and actual editable channels. Preserve chemical/representation changes
and sampled timing uncertainty. rr_ references are contextual only; never use a future measurement
as current observable evidence or manufacture a coordinate derivative of affinity, SA or a category.
Retain necessary unsupported goals as design_only and explain their missing evaluator or control map.

1. Read goal_pools and biology_decisions. Scientific goals include necessary unresolved directions;
researchable directions and biological compilation candidates are separate. Review the original goal
even when the evaluator is unavailable; do not replace it with an unrelated easy coordinate proxy.
2. Retrieve knowledge by the goal's mechanism, observable and role. prepare_function_synthesis can
search by a mechanism query or inspect chosen catalog IDs. Identify original objects, coupled terms,
response, prerequisites and gradient targets; decide what is retained, specialized or reconstructed.
Read relevant workflow reference sections when useful; sources are not instructions or proof of
efficacy. A copied formula or a source-consistency test cannot supply this transfer decision.
3. Then determine mathematical target sets or justified optimization relations, chemical hypotheses,
observable support and scale origins. The biological goal says what needs repair; its mathematical
acceptable set says which relations count as repair. A reference value does not imply a percentage
tolerance and a screening threshold is not physical calibration. Leave unknown parameters explicit
or explain bounded pilot assumptions; no fixed window, objective count, norm or weights are supplied.
4. Derive local response before constructing: direction, active/stopping regions, curvature,
symmetries and coupled effects. Preserve periodic/signed geometry, typed interaction behavior or
complete ready physical energy where relevant. A proposed multi-variable potential must produce
the declared response; a constrained direction need not be disguised as a scalar potential.
Use record_function_derivation for partial transfer/target/response/candidate summaries. It is editable
memory and decision help, never a mandatory tool sequence or extra gate.
5. Construct candidates from that reasoning. construct_direction_potential serializes expert-chosen
relations, parameters and within-mechanism operators, with actual lineage and derivative examples.
Its shape helpers are compilation options, not the universe of reasonable mathematics. Custom
constructs can use stage_mathematical_direction; unsupported evaluator needs stay design_only.
Compose only independent necessary goals; derive joint acceptable-set semantics and marginal
response. Rank alone never supplies a coefficient or stage. derive_allocation_operator is optional
calculation AFTER a justified response choice, not a source of biological importance.
6. Compare actual copy-coordinate response with compare_constructed_architectures: duplication,
inactive unresolved goals, coupling/conflict, projected norms and preservation effects. Revise the
construction using those measurements. A copy gradient is not a live pullback or future outcome;
mixed representations need an actual shared derivative map, and first-order preservation is limited.
7. Once ready, run the existing full-design numerical/binding test and submit the exact artifact.
Deferred designs can be submitted explicitly without claiming numerical approval. Use targeted
patches after specific final-contract feedback; partial planning records do not consume repair budget.

Keep every selected scientific goal and preservation direction represented. Missing force-field,
surface, charge, categorical or oracle inputs remain explicit. Research may justify request_biology_revision
with concrete evidence/scope changes; it cannot silently activate a deferred biological goal or omit
one. A whole energy, area or field cannot be attributed to a pair-distance proxy. Population weights
and black-box scores retain their roles unless a verified estimator enables another path. Record new
function/aggregation derivations separately from original source formulas. Exact schema/grammar are
available on demand; generic background reference eligibility rules concern actual execution, not
selection of scientific goals. No additional workflow gate is introduced. Guidance strength remains
external; unrun live/terminal validation stays not_run rather than a claimed acceptance condition.'''


def run_experts(runtime, state):
    materials = state.get('workflow_materials', {})
    instructions_by_role = {}
    for role, text in [('biology', BIOLOGY_INSTRUCTIONS), ('mathematics', MATH_INSTRUCTIONS)]:
        reference = initial_role_reference(materials, role)
        instructions_by_role[role] = text + ('\n\nGeneric background reference; apply under the current-state task and assigned role:\n' + reference['text'] if reference else '')
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
    raw_summary = raw_reference_summary(raw_context)
    raw_analysis = state.get('raw_reference_analysis', {}) if raw_context['status'] in ('available', 'partial') else {}
    current = current_state_context(packet, report, state['model_dynamics'], raw_context)

    def invoke(role, tools, context, completed):
        run_tools(runtime._model('molthinker.'+role), tools, instructions=instructions_by_role[role],
                  context=context, state=state, node='thinker.'+role,
                  max_steps=runtime.config.runtime.max_agent_steps,
                  max_repairs=runtime.config.runtime.max_repairs, completed=completed)

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
        def get_expert_contract() -> dict:
            """Read the BiologyPlan JSON schema before submitting a plan."""
            return {'schema': BiologyPlan.model_json_schema(), 'factors': list(FACTORS),
                    'audit_required': runtime.config.thinker.require_design_audit}

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
            if runtime.config.thinker.require_design_audit and inspected_factors != set(FACTORS):
                from .contract_errors import ContractValidationError
                raise ContractValidationError('Assess every biophysical factor before selecting a reward', ['factor_assessment'])
            bio['plan'] = validate_biology(plan, report, packet,
                require_audit=runtime.config.thinker.require_design_audit)
            return {'status':'accepted', 'biology_plan':deepcopy(bio['plan'])}

        invoke('biology', [get_expert_contract, inspect_biophysical_context, record_biology_decision, read_workflow_reference, list_measurement_references, inspect_measurement, inspect_measurements, *temporal_tools, service.tool_for('biology'), submit_biology_plan],
               {'report':bounded_values(report), 'model_dynamics':state['model_dynamics'],
                'monitor_feedback':state.get('monitor_event', {}), 'revision_request':feedback,
                'current_state':current, 'decision_workspace':BIOLOGY_GUIDE,
                'raw_reference':raw_summary, 'raw_reference_analysis':deepcopy(raw_analysis),
                'workflow_references':reference_catalog(materials),
                'previous_plan':state.get('biology_plan'), 'discussion_round':discussion}, lambda:bool(bio))
        state['biology_plan'] = bio['plan']
        biology = bio['plan']
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
        staged_directions = {}
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
        def get_expert_contract() -> dict:
            """Read mathematical schema, observable bindings and expression grammar before designing."""
            return {'schema':MathematicalDesign.model_json_schema(), 'observables':OBSERVABLES,
                    'audit_required': runtime.config.thinker.require_design_audit,
                    'observable_fields':['observable_id','kind','view','atom_ids','evidence_ids','parameters'],
                    'parameters':{'receptor_distance':['receptor_serial','residue_id'],
                                  'anchor_offset':['reference (three-vector)','origin'],
                                  'direction_alignment':['reference (three-vector)','origin'],
                                  'others':[]},
                    'expression':{'leaf':{'op':'observable','id':'observable_id'},
                                  'constant':{'op':'constant','value':1.0,'unit':'angstrom','origin':'evidence or declared scale origin'},
                                  'operator':{'op':'subtract','args':['expression','expression']},
                                  'unary':['relu','abs','sqrt','sin','cos','power'],
                                  'binary':['add','subtract','multiply','divide','maximum','minimum','periodic_difference'],
                                  'reduce':['sum','mean'], 'power_extra_field':'exponent in [0.5,8]',
                                  'units':list(UNITS), 'limits':'128 nodes, depth 12; scalar nonnegative dimensionless output'},
                    'strategy_examples':[{'mode':'scalar_potential','aggregation':{'op':'single'},'justification':'Case-specific reason required'},
                                         {'mode':'common_descent','aggregation':None,'justification':'Case-specific reason required'},
                                         {'mode':'design_only','aggregation':None,'justification':'Retain unresolved function/controller needs without execution'}],
                    'state_context':{'representations':bounded_values(packet['representations']),
                                     'chemical_readiness':packet['steering']['chemical_readiness'],
                                     'receptor_atom_count':len(packet['steering']['receptor_atoms']),
                                     'support_rule':'Use exact localized measurement references for atom/receptor bindings; full receptor coordinates are retained by the Executor, not repeated in this contract.',
                                     'graph_signatures':packet['steering']['graph_signatures']}}

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
                'biology_direction': research_map[direction_id],
                'derivation_record':deepcopy(workspace['decisions']['mathematics'].get(direction_id, {})),
                'next_questions': {name: MATH_GUIDE['stages'][name] for name in ('source_transfer','target_sets','local_response')},
                'scope':'Actual sources and optional derivation help. Source availability does not activate a biological goal or supply its target parameters.'}
            workspace['source_transfers'].append(deepcopy(result))
            return result

        @tool
        def construct_direction_potential(direction_id: str, relations: list[dict[str, Any]],
                                           within_direction: str, interpretation: dict[str, Any] | None = None) -> dict:
            """Construct and retain a draft from selected relation shapes/targets, with exact sources and derivative examples."""
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
            result['derivation_record'] = deepcopy(workspace['decisions']['mathematics'].get(direction_id, {}))
            workspace['constructions'].append(deepcopy(result))
            return result

        @tool
        def compare_constructed_architectures(strategies: list[dict[str, Any]]) -> dict:
            """Measure alternative draft controllers on coordinate copies; report allocation/conflict without choosing or gating."""
            result = preview_architectures(packet, biology, staged_directions, state['model_dynamics'], strategies)
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
        def request_biology_revision(direction_ids: list[str], issue: str, requested_change: str) -> dict:
            """Return evidence gaps, infeasible priorities or conflicting requirements to biology (bounded rounds)."""
            if math_result or not direction_ids or not set(direction_ids) <= set(research_map) or min(len(issue),len(requested_change)) < 12:
                raise ValueError('Revision must identify selected directions and a concrete issue/change')
            math_result['revision'] = dict(direction_ids=direction_ids, issue=issue, requested_change=requested_change)
            return {'status':'revision_requested', **math_result['revision']}

        @tool(args_schema=mathematical_draft_tool_schema())
        def test_mathematical_design(design: dict[str, Any]) -> dict:
            """Check a proposed handoff and actually test its local expressions and controller on coordinate copies."""
            if not isinstance(design,(dict,MathematicalDesign)):
                raise ValueError('Propose a complete mathematical draft before patching')
            proposed_draft['design']=deepcopy(design.model_dump() if isinstance(design,MathematicalDesign) else design)
            checked=validate_math(design,biology,packet,state['retrieval_records'],service.sources(), report=report,
                require_audit=runtime.config.thinker.require_design_audit)
            spec,deferred=compile_expert_spec(packet,report,biology,checked,state['retrieval_records'],state['model_dynamics'],service.sources())
            if deferred:
                tested_draft['design']=deepcopy(checked)
                return deferred
            from .executor import validate_and_test_reward
            validation=validate_and_test_reward(packet,spec,report)
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
            return stage_direction_draft(staged_directions, direction, targets)

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
                require_audit=runtime.config.thinker.require_design_audit)
            spec,deferred=compile_expert_spec(packet,report,biology,checked,state['retrieval_records'],state['model_dynamics'],service.sources())
            if not deferred and digest(checked) not in tested_designs:
                raise ValueError('test_mathematical_design must pass for the exact submitted artifact')
            math_result['design'] = checked
            return {'status':'accepted', 'mathematical_design':deepcopy(math_result['design'])}

        math_tools = [get_expert_contract, get_function_catalog, prepare_function_synthesis, record_function_derivation, read_workflow_reference, *temporal_tools, construct_direction_potential,
                               compare_constructed_architectures, derive_allocation_operator, inspect_biophysical_context, list_measurement_references, inspect_measurement, inspect_measurements,
                               search_direction_knowledge, inspect_direction_functions, service.tool_for('mathematics'),
                               request_biology_revision, stage_mathematical_direction, test_staged_mathematical_design,
                               patch_and_test_mathematical_design, submit_mathematical_design]
        if len(targets)==1:
            math_tools.append(test_mathematical_design)
        invoke('mathematics', math_tools,
               {'biology_plan':biology, 'report':bounded_values(report), 'model_dynamics':state['model_dynamics'],
                'current_state':current, 'goal_pools':pools,
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
            if discussion == runtime.config.thinker.max_discussions:
                return {'deferral':{'status':'design_only', 'reason':'Expert discussion budget exhausted', 'revision':feedback}}
            continue
        state['mathematical_design'] = math_result['design']
        spec, deferral = compile_expert_spec(packet, report, biology, math_result['design'],
                                             state['retrieval_records'], state['model_dynamics'], service.sources())
        append_trace(state, node='thinker', kind='decision', summary='Biology priorities and mathematical design bound to evidence',
                     output={'biology_plan':biology, 'mathematical_design':math_result['design'], 'deferral':deferral})
        return {'deferral':deferral} if deferral else {'spec':spec}
    raise RuntimeError('Unreachable expert scheduling state')
