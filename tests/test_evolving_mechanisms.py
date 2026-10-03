"""Current-chemistry control, ranked response and compact expert collaboration."""
import copy
import json
from pathlib import Path

import pytest
import torch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rdkit import Chem

from test_expert_system import case, make_models, observations
from test_design_audit import audited
from test_constructive_experts import param, retrieval
from molsteer.agents.config import load_config
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.priority import allocate_priorities
from molsteer.agents.expert_context import compact_history, task_context, recovery_candidate, draft_index, tool_receipt
from molsteer.agents.expert_contracts import validate_math, compile_expert_spec, ModelDynamicsContext
from molsteer.agents.reward_synthesis import construct_direction
from molsteer.molexecutor.expert_control import ExpertEvaluator, control_direction, run_expert_trial
from molsteer.molexecutor.chemical_references import geometry_reference, ChemicalReferenceUnavailable
from molsteer.molthinker.expressions import observable_value
from molsteer.molmonitor.checks import gradient_check


def typed_bond(coords, smiles, diagnostics=None):
    obs = dict(observable_id='stretch', kind='bond_length_error', atom_ids=[0, 1], parameters={})
    return observable_value(obs, coords, [0, 1], {}, molecule=Chem.MolFromSmiles(smiles), diagnostics=diagnostics)


def test_current_bond_type_changes_reference_and_force_without_initial_graph_lock():
    x = torch.tensor([[0., 0., 0.], [1.8, 0., 0.]], dtype=torch.float64, requires_grad=True)
    bindings = {}
    cc = typed_bond(x, 'CC', bindings)
    cc_reference = bindings['stretch']['reference']
    ic = typed_bond(x, 'IC', bindings)
    assert cc > 0 and ic < 0 and cc_reference < 1.8 < bindings['stretch']['reference']
    cc_gradient, = torch.autograd.grad(cc.square(), x, retain_graph=True)
    ic_gradient, = torch.autograd.grad(ic.square(), x)
    assert cc_gradient[1, 0] > 0 > ic_gradient[1, 0]
    assert gradient_check(lambda y: typed_bond(y, 'IC').square(), x.detach())['passed']
    # A disappearing relation is recorded separately, never called a repaired bond.
    inactive = typed_bond(x, 'C.C', bindings)
    assert inactive == 0 and bindings['stretch']['status'] == 'relation_absent'
    with pytest.raises(ChemicalReferenceUnavailable):
        geometry_reference(None, 'bond_length_error', [0, 1])


def test_current_angle_reference_and_typed_exclusion_have_coordinate_derivatives():
    x = torch.tensor([[0., 0., 0.], [1.4, .1, 0.], [2.1, 1.2, .1]], dtype=torch.float64)
    obs = dict(observable_id='bend', kind='bond_angle_error', atom_ids=[0, 1, 2], parameters={})
    values = [observable_value(obs, x, [0, 1, 2], {}, molecule=Chem.MolFromSmiles(s)) for s in ('CCC', 'C=CC')]
    assert abs(float(values[0]-values[1])) > .1
    assert gradient_check(lambda y: observable_value(obs, y, [0, 1, 2], {}, molecule=Chem.MolFromSmiles('C=CC')).square(), x)['passed']
    packet = {'steering': {'receptor_atoms': [dict(serial=8, residue_id='A:12', element='C', coords=[0., 0., 0.])]}}
    obs = dict(observable_id='overlap', kind='typed_steric_overlap', atom_ids=[0],
               parameters=dict(receptor_serial=8, residue_id='A:12', buffer_ratio=1.))
    y = torch.tensor([[3.5, 0., 0.]], dtype=torch.float64)
    assert observable_value(obs, y, [0], packet, elements=['C']) == 0
    assert observable_value(obs, y, [0], packet, elements=['I']) > 0
    assert gradient_check(lambda z: observable_value(obs, z, [0], packet, elements=['I']).square(), y)['passed']


def test_rank_coefficients_change_actual_gradient_and_explicit_allocation_is_preserved():
    biology = {'directions': [dict(direction_id=k, rank=i+1, disposition='optimize') for i, k in enumerate(('a', 'b'))]}
    design = {'directions': [dict(direction_id=k, status='executable') for k in ('a', 'b')],
              'strategy': dict(mode='scalar_potential', aggregation={'op': 'weighted_sum'}, justification='Independent normalized repairs')}
    allocated = allocate_priorities(design, biology, decay=.25)
    assert allocated['strategy']['priority_weights'] == {'a': 1., 'b': .25}
    x = torch.tensor([1., 1.], requires_grad=True)
    costs = {'a': x[0].square(), 'b': x[1].square()}
    direction, _, _ = control_direction(costs, x, torch.ones_like(x), allocated['strategy'])
    assert torch.allclose(direction, torch.tensor([-2., -.5]))
    biology['directions'].reverse()
    for i, d in enumerate(biology['directions']): d['rank'] = i+1
    reverse = allocate_priorities(design, biology, decay=.25)
    direction, _, _ = control_direction(costs, x, torch.ones_like(x), reverse['strategy'])
    assert torch.allclose(direction, torch.tensor([-.5, -2.]))
    design['strategy']['priority_weights'] = {'a': .7, 'b': 1.}
    assert allocate_priorities(design, biology)['strategy']['priority_weights'] == {'a': .7, 'b': 1.}


def test_partial_strategy_returns_correctable_contract_feedback():
    from molsteer.agents.contract_errors import ContractValidationError
    from molsteer.agents.loop import _validation_feedback
    design = {'directions': [], 'strategy': {'aggregation': {'op': 'single'}}}
    with pytest.raises(ContractValidationError) as error:
        allocate_priorities(design, {'directions': []})
    feedback = _validation_feedback(error.value)
    assert feedback['validation_path'] == ['strategy']
    assert 'mode, aggregation and justification' in feedback['validation_hint']
    assert design['strategy'] == {'aggregation': {'op': 'single'}}


def test_legacy_constraint_analysis_cannot_suspend_scalar_objectives():
    evaluator = object.__new__(ExpertEvaluator)
    evaluator.roles = {'repair': 'optimize', 'independent': 'optimize', 'preserve': 'constraint'}
    evaluator.dependencies = {'repair': ['preserve'], 'independent': []}
    evaluator.constraint_modes = {'preserve': 'native_nonincrease'}
    baseline = {'preserve': torch.tensor(.2)}
    assert evaluator.constraints({'preserve': torch.tensor(.15)}, baseline) == []
    assert evaluator.constraints({'preserve': torch.tensor(.21)}, baseline) == ['preserve']
    evaluator.constraint_modes['preserve'] = 'absolute'
    assert evaluator.constraints({'preserve': torch.tensor(.15)}, baseline) == ['preserve']
    evaluator.directions=[{'direction_id':'repair'},{'direction_id':'independent'}]
    assert set(evaluator.objectives({'repair':torch.tensor(1.),'independent':torch.tensor(2.)}))=={'repair','independent'}
    with pytest.raises(ValueError,match='components missing'):
        evaluator.objectives({'independent':torch.tensor(2.)})


def test_common_descent_preserves_rank_preference_and_resolves_first_order_conflict():
    from molsteer.agents.optimization import priority_descent
    x = torch.tensor([1., 1.], dtype=torch.float64, requires_grad=True)
    strategy = dict(mode='common_descent', aggregation=None, priority_weights={'a': 1., 'b': .25})
    direction, _, _ = control_direction({'a': x[0].square(), 'b': x[1].square()}, x, torch.ones_like(x), strategy)
    assert float(direction[0]/direction[1]) == pytest.approx(4.)
    # A preferred direction that would increase the second goal is projected.
    result = priority_descent([[1., 0.], [-1., 1.]], [1., 1.], [1., .1])
    assert result['status'] == 'candidate_descent'
    assert max(result['directional_derivatives']) <= 1e-9
    assert min(result['directional_derivatives']) < -.1
    blocked = priority_descent([[1., 0.], [-1., 0.]], [1., 1.], [1., .1])
    assert blocked['status'] == 'pareto_stationary_or_inactive'


def test_missing_typed_component_fails_instead_of_silently_shrinking_reward(case):
    packet = case[0]
    distance = copy.deepcopy(case[3]['directions'][0]['observables'][0])
    typed = dict(distance, kind='bond_length_error', observable_id='typed')
    unit = {'op': 'constant', 'value': 1., 'unit': 'angstrom', 'origin': 'Synthetic normalization'}
    def term(ident, obs):
        z = {'op': 'observable', 'id': obs['observable_id']}
        return dict(direction_id=ident, status='executable', observables=[obs],
            expression={'op': 'power', 'exponent': 2, 'args': [{'op': 'divide', 'args': [z, unit]}]})
    spec = dict(biology_plan={'directions': [dict(direction_id=k, disposition='optimize', preservation_conditions=[]) for k in ('typed', 'plain')]},
        mathematical_design={'directions': [term('typed', typed), term('plain', distance)], 'strategy': {'mode': 'scalar_potential', 'aggregation': {'op': 'weighted_sum'}}})
    evaluator = ExpertEvaluator(spec, packet)
    view = distance['view']
    coords = {view: torch.tensor(packet['steering']['coordinate_snapshots'][view]['coords_angstrom'], requires_grad=True)}
    with pytest.raises(ValueError,match='Reward component unavailable: typed'):
        evaluator.components(coords, molecules={view: None})
    assert set(evaluator.unavailable)=={'typed'}


def test_current_reference_shape_passes_existing_audit_without_graph_freeze(case):
    packet, report, biology, design, _ = audited(case)
    goal = next(d for d in biology['directions'] if d['direction_id'] == 'repair')
    obs = copy.deepcopy(design['directions'][0]['observables'][0])
    obs['kind'] = 'bond_length_error'
    goal['repair_clauses'][0]['observable_kind'] = obs['kind']
    ret = retrieval(case)
    relation = dict(observable=obs, function_id='G01', shape='current_reference_quadratic',
                    parameters={'scale': param(.2, role='normalization')}, clause_ids=['distance_window'])
    result = construct_direction(goal, [relation], 'single', [ret], packet)
    assert result['status'] == 'draft_only', result
    design['directions'] = [result['direction']]
    design['design_audit']['graph_policy'] = 'current_chemistry'
    checked = validate_math(design, biology, packet, [ret], {}, report=report, require_audit=True)
    assert not {'lower', 'upper', 'target'} & {r['origin'] for r in checked['directions'][0]['reference_parameters']}
    spec, deferred = compile_expert_spec(packet, report, biology, checked, [ret], ModelDynamicsContext().model_dump(), {})
    assert deferred is None
    trial = run_expert_trial(packet, spec)
    assert trial['numerical_gradient']['passed']


def test_compact_context_has_one_recovery_candidate_and_drops_oversized_mutation_round():
    candidate = {'directions': [{'direction_id': 'a', 'formula': 'obsolete_expression'*1000}]}
    event = {'kind': 'recovery', 'recovery': {'previous_attempt': 3,
        'mathematical_direction_candidates': {'a': {'formula': 'older_expression'}}, 'full_model_authored_candidate': candidate}}
    assert 'obsolete_expression' not in json.dumps(task_context(event))
    assert recovery_candidate(event) == candidate
    assert 'formula' not in draft_index({'a': candidate['directions'][0]})['a']
    prefix = [SystemMessage(content='rules'), HumanMessage(content='context')]
    mutation = AIMessage(content='', tool_calls=[dict(name='test_mathematical_design', args={'design': candidate}, id='old')])
    source = AIMessage(content='', tool_calls=[dict(name='inspect_direction_functions', args={}, id='source')],
                       additional_kwargs={'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'signed_provider_token'}]})
    messages = [*prefix, mutation, ToolMessage(content='old failure', tool_call_id='old', name='test_mathematical_design'),
                source, ToolMessage(content='needed source', tool_call_id='source', name='inspect_direction_functions')]
    compact = compact_history(messages, {'draft_index': {'a': 'current'}}, max_chars=4000, recent_rounds=2)
    serialized = json.dumps([m.model_dump() for m in compact])
    assert 'obsolete_expression' not in serialized and 'old failure' not in serialized
    assert 'needed source' in serialized and 'signed_provider_token' in serialized
    calls = [c['id'] for m in compact for c in getattr(m, 'tool_calls', [])]
    responses = [m.tool_call_id for m in compact if isinstance(m, ToolMessage)]
    assert calls == responses == ['source']
    # An oversized source round is dropped as a pair, and stays retrievable by receipt.
    messages[-1] = ToolMessage(content='x'*5000, tool_call_id='source', name='inspect_direction_functions')
    compact = compact_history(messages, {'recent_observations': [{'event_id': 'source_receipt'}]}, max_chars=2000, recent_rounds=2)
    assert len(compact) == 3 and 'source_receipt' in compact[-1].content
    receipt = tool_receipt({'event_id': 'construction', 'tool_name': 'construct_direction_potential',
        'output': {'status': 'needs_input', 'hint': 'Select the current-reference observable.', 'direction': candidate}})
    assert receipt['feedback']['hint'] == 'Select the current-reference observable.'
    assert 'direction' in receipt['output_fields'] and 'obsolete_expression' not in json.dumps(receipt)
    assert 'hint' not in tool_receipt({'output': {'status': 'error', 'hint': 'Superseded correction'}},
                                    include_feedback=False)['feedback']


def test_configured_long_history_retains_source_rounds_under_character_budget():
    cfg = load_config()
    cfg.thinker.history_recent_rounds = 32
    cfg.thinker.history_max_chars = 100000
    messages = [SystemMessage(content='rules'), HumanMessage(content='context')]
    for index in range(12):
        call_id = f'source_{index}'
        messages.extend([
            AIMessage(content='', tool_calls=[dict(name='inspect_measurement', args={}, id=call_id)]),
            ToolMessage(content=f'bound evidence {index}', tool_call_id=call_id, name='inspect_measurement'),
        ])
    compact = compact_history(messages, {}, max_chars=cfg.thinker.history_max_chars,
                              recent_rounds=cfg.thinker.history_recent_rounds)
    assert [m.tool_call_id for m in compact if isinstance(m, ToolMessage)] == [f'source_{i}' for i in range(12)]
    cfg.thinker.history_max_chars = 2000
    compact = compact_history(messages, {}, max_chars=cfg.thinker.history_max_chars,
                              recent_rounds=cfg.thinker.history_recent_rounds)
    calls = [c['id'] for m in compact for c in getattr(m, 'tool_calls', [])]
    responses = [m.tool_call_id for m in compact if isinstance(m, ToolMessage)]
    assert calls == responses and len(responses) < 12


def test_recent_draft_and_validation_rounds_survive_compaction_within_budget():
    prefix = [SystemMessage(content='rules'), HumanMessage(content='context')]
    staged = AIMessage(content='', tool_calls=[dict(name='stage_mathematical_direction',
        args={'direction': {'direction_id': 'local', 'formula': 'current candidate'}}, id='stage')],
        additional_kwargs={'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'provider_signature'}]})
    tested = AIMessage(content='', tool_calls=[dict(name='test_staged_mathematical_design',
        args={'strategy': {'mode': 'single'}}, id='test')])
    messages = [*prefix, staged,
        ToolMessage(content='draft_only; saved, not tested', tool_call_id='stage'),
        tested, ToolMessage(content='specific numerical feedback', tool_call_id='test')]
    compact = compact_history(messages, {'draft_index': {'local': 'current'}},
                              max_chars=10000, recent_rounds=2)
    assert compact[3] is staged and compact[5] is tested
    assert [m.tool_call_id for m in compact if isinstance(m, ToolMessage)] == ['stage', 'test']
    assert 'current candidate' in json.dumps([m.model_dump() for m in compact])
    # A later budget reduction removes whole pairs; it never retains orphan replies.
    compact = compact_history(compact, {}, max_chars=1, recent_rounds=2)
    assert len(compact) == 3


def test_mathematical_conflict_revisits_biology_and_invalidates_changed_draft(case):
    packet, report, biology, design, _ = case
    cfg = load_config(); cfg.thinker.require_design_audit = False; cfg.thinker.external_research = False
    cfg.runtime.max_repairs = 0; cfg.agents['molmonitor'].enabled = False
    cfg.runtime.trace_dir = Path('outputs/test_evolving_mechanisms')
    models = make_models(case); seen = {}
    def bio(n, messages):
        plan = copy.deepcopy(biology)
        if n == 2:
            context = json.loads(messages[1].content)
            seen['request'] = context['revision_request']
            goal = next(d for d in plan['directions'] if d['direction_id'] == 'repair')
            goal['chemical_state'] = 'Current chemical hypothesis; initial types are not immutable.'
            goal['priority_reason'] = 'Retain the necessary repair under the revised chemical interpretation.'
        return [('submit_biology_plan', {'plan': plan})]
    def math(n, messages):
        if n == 1: return [('stage_mathematical_direction', {'direction': design['directions'][0]})]
        if n == 2:
            return [('request_biology_revision', dict(direction_ids=['repair'],
                issue='An initial chemical-type assumption conflicts with the observed transition.',
                requested_change='Reinterpret the local mechanism under current chemistry and reassess priority.',
                evidence_ids=biology['directions'][0]['evidence_ids'], proposed_changes={'chemical_state': 'current chemistry'}))]
        if n == 3:
            seen['math_context'] = json.loads(messages[1].content)
            return [('read_expert_workspace', {'section': 'drafts'}), ('search_direction_knowledge', {'direction_id': 'repair', 'query': 'flat-bottom local_geometry'})]
        if n == 4:
            seen['drafts'] = observations(messages, 'read_expert_workspace')[0]['value']
            d = copy.deepcopy(design)
            d['directions'][0]['retrieval_ids'] = [observations(messages, 'search_direction_knowledge')[0]['retrieval_id']]
            return [('test_mathematical_design', {'design': d})]
        return [('submit_mathematical_design', {})]
    models['molthinker.biology'].respond = bio
    models['molthinker.mathematics'].respond = math
    state = AgentRuntime(cfg, models=models).run(packet, report, run_id='revision_current_chemistry')
    assert state['status'] == 'validated', state['trace'][-5:]
    assert seen['request']['proposed_changes'] == {'chemical_state': 'current chemistry'}
    assert seen['drafts'] == {}
    assert 'Current chemical hypothesis' in next(d for d in seen['math_context']['biology_plan']['directions'] if d['direction_id'] == 'repair')['chemical_state']
    assert state['expert_workspace_history'][1]['revision_resolution']['changed_direction_ids'] == ['repair']
    assert 'goal_pools' not in seen['math_context'] and 'raw_reference' not in seen['math_context']['current_state']
    assert 'PRIVATE_REASONING_NOT_FOR_AUDIT' not in json.dumps(state)
