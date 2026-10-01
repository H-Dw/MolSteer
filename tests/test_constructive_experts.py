"""Forward design assistance, numerical transfer, and non-gating integration."""
import copy
import json
import math
from pathlib import Path
import pytest
import torch
from test_expert_system import case, make_models
from test_design_audit import audited
from molsteer.agents.config import load_config
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.expert_contracts import validate_math
from molsteer.agents.decision_workspace import new_workspace, update_workspace
from molsteer.agents.reward_synthesis import construct_direction, function_card, preview_architectures, _build_relation, derive_allocation_response
from molsteer.molthinker.expressions import evaluate_expression
from molsteer.molthinker.expressions import aggregate_objectives
from molsteer.molthinker.research.corpus import MarkdownCorpus


def test_declared_independent_defects_need_two_goals_without_outcome_priors():
    workspace = new_workspace()
    candidates = {'goals': [{'goal_id':'geometry', 'covers':['bond']},
                            {'goal_id':'stability', 'covers':['distributed_strain']}]}
    before = copy.deepcopy(candidates)
    update_workspace(workspace, 'biology', 'candidate_goals', candidates)
    one = update_workspace(workspace, 'biology', 'selection',
        {'selected_goal_ids':['geometry'], 'necessary_mechanism_ids':['bond','distributed_strain']})
    assert one['coverage_review']['uncovered_mechanism_ids'] == ['distributed_strain']
    two = update_workspace(workspace, 'biology', 'selection', {'selected_goal_ids':['geometry','stability']})
    assert two['coverage_review']['uncovered_mechanism_ids'] == []
    assert all(x['newly_uncovered_mechanism_ids'] for x in two['coverage_review']['removal_review'])
    assert candidates == before and two['blocking'] is False
    assert 'selected_strategy' not in two and 'net_value' not in json.dumps(two)


def test_correlated_alerts_do_not_receive_independent_scores_or_forced_deletion():
    workspace = new_workspace()
    update_workspace(workspace, 'biology', 'candidate_goals', {'goals': [
        {'goal_id':'bond_proxy','covers':['one_deformation']},
        {'goal_id':'coupled_energy','covers':['one_deformation']}]})
    result = update_workspace(workspace, 'biology', 'selection',
        {'selected_goal_ids':['bond_proxy','coupled_energy'], 'necessary_mechanism_ids':['one_deformation']})
    assert all(not row['newly_uncovered_mechanism_ids'] for row in result['coverage_review']['removal_review'])
    assert result['coverage_review']['selected_goal_ids'] == ['bond_proxy','coupled_energy']
    assert 'weights' not in result and 'rank' not in result


def test_partial_workspace_is_nonblocking_and_does_not_invent_target_values():
    workspace = new_workspace()
    result = update_workspace(workspace, 'mathematics', 'local_response',
        {'direction':'Reduce a measured strain under an unresolved chemical hypothesis.'}, 'stability')
    assert result['status'] == 'workspace' and result['blocking'] is False
    assert 'target_sets' not in workspace['mathematics']['stability']
    unknown = update_workspace(workspace, 'biology', 'unknown', {})
    assert unknown['status'] == 'needs_input' and unknown['blocking'] is False


def test_missing_coverage_is_unknown_instead_of_a_satisfied_empty_set():
    workspace = new_workspace()
    result = update_workspace(workspace, 'biology','selection', {'selected_goal_ids':['unresolved_goal']})
    review = result['coverage_review']
    assert review['necessary_mechanism_ids'] is None and review['uncovered_mechanism_ids'] is None
    assert review['coverage_information']=='partial'
    assert review['selected_goal_ids_without_coverage']==['unresolved_goal']
    assert review['removal_review'][0]['newly_uncovered_mechanism_ids'] is None
    assert result['blocking'] is False


def param(value, unit='angstrom', role='tolerance'):
    return dict(value=value, unit=unit, origin=f'Explicit synthetic {role} value {value} {unit}', role=role,
        provenance='declared_assumption', evidence_ids=[], source_locators=[],
        derivation='Uncalibrated synthetic test input; never an inferred physical tolerance.')


def relation(case):
    return dict(observable=copy.deepcopy(case[3]['directions'][0]['observables'][0]), function_id='G01',
        shape='interval_quadratic', parameters={'lower': param(1.3), 'upper': param(1.6), 'scale': param(.2, role='normalization')},
        clause_ids=['distance_window'])


def retrieval(case, ident='G01'):
    corpus = MarkdownCorpus(Path(__file__).resolve().parents[1]/'knowledge')
    result = corpus.search('Inspect function mechanism', function_id=ident)
    return dict(result, direction_id='repair', retrieval_id='ret_constructive_'+ident)


def test_constructed_interval_passes_existing_full_audit_and_keeps_packet_unchanged(case):
    p, r, b, d, _ = audited(case); before = copy.deepcopy(p)
    bio = next(x for x in b['directions'] if x['direction_id'] == 'repair'); ret = retrieval(case)
    result = construct_direction(bio, [relation(case)], 'single', [ret], p)
    assert result['status'] == 'draft_only' and result['validated'] is False
    d['directions'] = [result['direction']]
    checked = validate_math(d, b, p, [ret], {}, report=r, require_audit=True)
    expr = checked['directions'][0]['expression']
    for value, expected, sign in [(1.1, .5, -1), (1.45, 0, 0), (1.8, .5, 1)]:
        z = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        cost = evaluate_expression(expr, {'distance': z}); g, = torch.autograd.grad(cost, z)
        assert float(cost.detach()) == pytest.approx(expected)
        assert (float(g) > 0) - (float(g) < 0) == sign
    assert p == before


def test_circular_specialization_respects_wrap_and_source_role(case):
    ret = retrieval(case, 'G05'); source = ret['records'][0]
    spec = {'observable': {'observable_id': 'torsion', 'kind': 'dihedral'}, 'shape': 'periodic_cosine',
        'parameters': {'target': param(-math.pi+.05, 'radian', 'reference'), 'scale': param(.2, 'radian', 'normalization')}}
    part = _build_relation(spec, source)
    assert part['source_transform'] == 'new_surrogate'
    z = torch.tensor(math.pi-.05, dtype=torch.float64, requires_grad=True)
    loss = evaluate_expression(part['expression'], {'torsion': z}); g, = torch.autograd.grad(loss, z)
    expected = (1-math.cos(.1))/(1-math.cos(.2))
    assert float(loss.detach()) == pytest.approx(expected) and g < 0
    assert 'not the reported energy' in part['derivation']


def test_bad_construction_returns_advice_and_does_not_create_an_execution_exception(case):
    p, _, b, _, _ = case; bio = next(x for x in b['directions'] if x['direction_id'] == 'repair')
    invalid = relation(case); invalid['parameters']['scale']['value'] = 0
    result = construct_direction(bio, [invalid], 'single', [retrieval(case)], p)
    assert result['status'] == 'needs_input' and result['blocking'] is False and 'positive' in result['hint']
    assert 'direction' not in result


def test_function_cards_cover_all_source_roles_and_never_coerce_population_to_coordinates(case):
    corpus = MarkdownCorpus(Path(__file__).resolve().parents[1]/'knowledge')
    cards = {x['function_id']: function_card(x) for x in corpus.chunks.values() if x.get('function_id')}
    assert len(cards) == 21
    assert 'complete typed molecular energy' in cards['P01']['transfer_reasoning']
    assert cards['P06']['constructible_shapes'] == []
    assert cards['S02']['role'] == 'population_weight'
    assert 'does not justify a within-molecule' in cards['S02']['transfer_reasoning']


def test_composition_is_derived_from_response_and_checked_by_actual_executor_expression():
    result = derive_allocation_response(.6, .2, 2., 'Synthetic response choice keeps pressure on both unsatisfied targets.')
    strategy = result['candidate_strategy']
    assert strategy['aggregation']['p'] == pytest.approx(1+math.log(2)/math.log(3))
    values = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in [('a', .6), ('b', .2)]}
    total = aggregate_objectives(values, strategy)
    a, b = torch.autograd.grad(total, tuple(values.values()))
    assert float(a/b) == pytest.approx(2.) and b > 0
    assert derive_allocation_response(1., 1., 2., 'Unequal importance cannot be inferred from equal deficits.')['status'] == 'underdetermined'
    unsupported = derive_allocation_response(2., 1., 1024., 'Extreme response ratio requires a different supported control family.')
    assert unsupported['derived_p'] == 11 and unsupported['candidate_strategy'] is None


def test_preview_measures_allocation_instead_of_selecting_a_fixed_aggregator(case):
    p, _, b, _, _ = case; bio = next(x for x in b['directions'] if x['direction_id'] == 'repair')
    result = construct_direction(bio, [relation(case)], 'single', [retrieval(case)], p)
    strategies = [dict(mode='scalar_potential', aggregation={'op': 'single'}, justification='One independently sufficient local test target.'),
                  dict(mode='common_descent', aggregation=None, justification='Compare the real projected copy gradient direction.')]
    before = copy.deepcopy(p)
    preview = preview_architectures(p, b, {'repair': result['direction']}, {'editable_atom_ids': None}, strategies)
    assert preview['scope'] == 'coordinate_copy_same_view' and preview['selected_strategy'] is None
    assert preview['previews'][0]['marginal_allocation'] == {'repair': 1.0}
    assert all('masked_gradient_norms' in row for row in preview['previews'])
    assert p == before


def test_preview_exposes_maximum_starvation_and_opposing_copy_gradients(case):
    p, _, b, _, _ = copy.deepcopy(case)
    bio = next(x for x in b['directions'] if x['direction_id'] == 'repair')
    snap = p['steering']['coordinate_snapshots'][relation(case)['observable']['view']]
    coords = dict(zip(snap['atom_ids'], snap['coords_angstrom']))
    a, c = relation(case)['observable']['atom_ids']
    distance = math.dist(coords[a], coords[c])
    def make(ident, target):
        direction = copy.deepcopy(bio); direction['direction_id'] = ident
        spec = relation(case); spec.update(shape='center_quadratic', clause_ids=[])
        spec['parameters'] = {'target': param(target, role='reference'), 'scale': param(.2, role='normalization')}
        return direction, construct_direction(direction, [spec], 'single', [retrieval(case)], p)['direction']
    one, one_draft = make('large_gap', distance-.4)
    two, two_draft = make('small_gap', distance-.1)
    b['directions'] = [one, two]
    maximum = dict(mode='scalar_potential', aggregation={'op': 'maximum'}, justification='Inspect worst-target marginal allocation, not an assumed preferred strategy.')
    result = preview_architectures(p, b, {'large_gap': one_draft, 'small_gap': two_draft}, {}, [maximum])
    assert result['previews'][0]['marginal_allocation'] == {'large_gap': 1., 'small_gap': 0.}
    two, two_draft = make('opposite', distance+.4); b['directions'] = [one, two]
    common = dict(mode='common_descent', aggregation=None, justification='Inspect whether the same copy variable admits a common repair direction.')
    result = preview_architectures(p, b, {'large_gap': one_draft, 'opposite': two_draft}, {}, [common])
    assert result['previews'][0]['status'] == 'pareto_stationary_or_inactive'
    assert result['previews'][0]['copy_gradient_cosines'][0][1] == pytest.approx(-1)
    assert result['selected_strategy'] is None


@pytest.mark.parametrize('partial_input', [False, True])
def test_constructive_tools_finish_through_existing_audits_without_full_ast_resubmission(case, partial_input):
    p, r, b, d, _ = audited(case)
    cfg = load_config(); cfg.runtime.trace_dir = Path('outputs/test_constructive'); cfg.thinker.external_research = False
    cfg.runtime.max_repairs = 0  # Advisory needs_input never consumes execution repair budget.
    cfg.agents['molmonitor'].enabled = False
    models = make_models(case)
    def biology(n, messages):
        if n == 1:
            return [('record_biology_decision', {'stage':'integration', 'record':{'unresolved':['Coupled mechanism remains a current-state hypothesis.']}})]
        if n == 2:
            return [('inspect_biophysical_context', {'factor': 'all'})]
        return [('submit_biology_plan', {'plan': b})]
    def math_model(n, messages):
        if partial_input and n == 1:
            incomplete = relation(case); incomplete['parameters']['scale']['value'] = 0
            return [('construct_direction_potential', {'direction_id': 'repair', 'relations': [incomplete], 'within_direction': 'single'})]
        if partial_input:
            n -= 1
        if n == 1:
            return [('construct_direction_potential', {'direction_id': 'repair', 'relations': [relation(case)], 'within_direction': 'single'})]
        if n == 2:
            return [('compare_constructed_architectures', {'strategies': [d['strategy']]})]
        if n == 3:
            return [('test_staged_mathematical_design', {k: d[k] for k in ('strategy', 'conflict_assessment', 'independent_evaluation', 'design_audit')})]
        return [('submit_mathematical_design', {})]
    models['molthinker.biology'].respond = biology
    models['molthinker.mathematics'].respond = math_model
    state = AgentRuntime(cfg, models=models).run(p, r, run_id='constructive_flow')
    assert state['status'] == 'validated', state['trace'][-4:]
    assert not any(e['kind'] == 'error' for e in state['trace'])
    assert state['expert_workspace_history'][0]['decisions']['biology']['integration']['unresolved']
    assert models['molthinker.mathematics'].calls == 4+int(partial_input)
    submitted = next(e for e in state['trace'] if e.get('tool_name') == 'submit_mathematical_design')
    assert submitted['input'] == {}
    assert 'PRIVATE_REASONING_NOT_FOR_AUDIT' not in json.dumps(state)
