"""Behavioral checks for current-state decision help and unresolved scientific goals."""
import copy
import json
from pathlib import Path
import pytest
from test_expert_system import case, make_models, observations
from test_design_audit import audited
from test_constructive_experts import relation
from molsteer.agents.config import load_config
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.design_audit import biophysical_context
from molsteer.agents.decision_workspace import (current_state_context, goal_pools,
    new_workspace, update_workspace)
from molsteer.agents.expert_contracts import compile_expert_spec, validate_math, ModelDynamicsContext
from molsteer.agents.workflow_guidance import load_workflow_materials, read_reference


ROOT = Path(__file__).resolve().parents[1]


def test_current_views_stay_distinct_and_input_is_not_mutated(case):
    packet, report = case[:2]
    before = copy.deepcopy(packet)
    context = biophysical_context(packet)
    assert {m['view'] for f in context['factors'] for m in f['observations']} == {'state','prediction'}
    for view in ('state', 'prediction'):
        filtered = biophysical_context(packet, view=view)
        assert all(m['view'] == view for f in filtered['factors'] for m in f['observations'])
    current = current_state_context(packet, report, ModelDynamicsContext().model_dump())
    assert current['representations'] == packet['representations']
    assert {f['report_group'] for f in current['findings']} == {'findings','raw_state_findings'}
    assert packet == before


def test_selected_science_and_research_are_not_identical_to_executable_goals(case):
    biology = copy.deepcopy(case[2])
    repair = next(d for d in biology['directions'] if d['direction_id']=='repair')
    repair.update(disposition='deferred', required=True)
    pools = goal_pools(biology)
    assert pools['scientific_goals'] == [repair]
    assert pools['compilation_candidate_ids'] == []
    assert repair['direction_id'] in pools['researchable_goal_ids']
    assert pools['unresolved_selected_goal_ids'] == [repair['direction_id']]
    assert all(d['direction_id'] in pools['researchable_goal_ids'] for d in biology['directions'])


def test_missing_scientific_goal_cannot_be_silently_omitted(case):
    packet, report, biology, design, ret = copy.deepcopy(case)
    extra = next(d for d in biology['directions'] if d['direction_id'] != 'repair')
    extra.update(disposition='deferred', required=True)
    with pytest.raises(ValueError, match='account for each selected'):
        validate_math(design, biology, packet, [ret], {})


def test_design_only_controller_preserves_ready_local_formula_without_execution(case):
    packet, report, biology, design, ret = copy.deepcopy(case)
    design['strategy'] = dict(mode='design_only', aggregation=None,
                             justification='Local component can be evaluated but the scientifically needed controller is unresolved.')
    spec, deferred = compile_expert_spec(packet, report, biology, design, [ret], ModelDynamicsContext().model_dump(), {})
    assert spec is None and deferred['status'] == 'design_only'
    assert design['directions'][0]['status'] == 'executable'


def test_references_are_actual_readable_sources_and_never_arbitrary_paths(tmp_path):
    root = tmp_path / 'workflow'; root.mkdir()
    (tmp_path/'outside.md').write_text('OUTSIDE_PRIVATE_CONTENT', encoding='utf-8')
    (root/'SKILL.md').write_text('# Workflow\n\n[local](reference.md) [outside](../outside.md)\n', encoding='utf-8')
    (root/'reference.md').write_text('# Reference\n\n## Mechanism\nActual supplied mechanism material.\n', encoding='utf-8')
    materials = load_workflow_materials(root/'SKILL.md')
    assert set(materials) == {'SKILL.md','reference.md'}
    assert read_reference(materials,'reference.md','Mechanism')['text'].endswith('Actual supplied mechanism material.')
    missing = read_reference(materials,'../outside.md')
    assert missing['status'] == 'needs_input' and missing['blocking'] is False
    assert 'OUTSIDE_PRIVATE_CONTENT' not in json.dumps(materials)


def test_public_workspace_redacts_private_fields_and_can_merge_partial_records():
    workspace = new_workspace()
    draft = {'target_relation':'Reference pending current-state measurement.',
             'private_reasoning':'PRIVATE_REASONING_PAYLOAD', 'api_key':'SECRET_PAYLOAD'}
    before = copy.deepcopy(draft)
    update_workspace(workspace, 'mathematics','target_sets', draft, 'repair')
    update_workspace(workspace, 'mathematics','target_sets', {'unknowns':['Tolerance uncalibrated.']}, 'repair')
    assert 'PRIVATE_REASONING_PAYLOAD' not in json.dumps(workspace)
    assert 'SECRET_PAYLOAD' not in json.dumps(workspace)
    assert workspace['mathematics']['repair']['target_sets']['target_relation'] == draft['target_relation']
    assert draft == before


def test_real_role_inputs_receive_current_state_and_references_without_forecast_tool(case):
    cfg = load_config(); cfg.thinker.require_design_audit=False; cfg.thinker.external_research=False
    cfg.agents['molmonitor'].enabled=False; cfg.runtime.trace_dir=Path('outputs/test_forward')
    models=make_models(case)
    originals={role:models['molthinker.'+role].respond for role in ('biology','mathematics')}
    seen={}
    for role in ('biology','mathematics'):
        def capture(n, messages, role=role):
            if n==1:
                seen[role]=(messages[0].content,json.loads(messages[1].content))
            return originals[role](n,messages)
        models['molthinker.'+role].respond=capture
    state=AgentRuntime(cfg,models=models).run(*case[:2],run_id='forward_input_fixture')
    assert state['status']=='validated', state['trace'][-4:]
    assert 'compare_intervention_opportunities' not in models['molthinker.biology'].tools
    for role,(prompt,context) in seen.items():
        assert 'terminal_gain' not in prompt and 'counterfactual terminal benefit' not in prompt
        assert 'current_state' in context and context['workflow_references']
        assert 'opportunity_workspace' not in context and 'opportunity_comparisons' not in context
    math_context=seen['mathematics'][1]
    assert math_context['goal_pools']['scientific_goals'][0]['direction_id']=='repair'
    assert math_context['biology_decisions']=={}  # Optional helpers did not invent a biological decision.
    assert state['expert_guidance_sources']['mathematics']['reference_id']=='references/knowledge-guided-composition.md'
    assert 'workflow_materials' not in state
    checkpoint=json.loads(Path(state['checkpoint_path']).read_text(encoding='utf-8'))
    assert checkpoint['artifacts']['workflow_reference_digests']==state['workflow_reference_digests']


def test_unimplemented_required_goal_reaches_source_transfer_and_is_retained(case):
    packet,report,biology,design,_=audited(case)
    repair=next(d for d in biology['directions'] if d['direction_id']=='repair')
    repair.update(disposition='deferred',required=True)
    for factor in biology['factor_assessment']:
        if 'repair' in factor['direction_ids']:
            factor.update(disposition='deferred',missing_requirements=['Chemical applicability of the declared reference remains unresolved.'])
    cfg=load_config(); cfg.thinker.external_research=False; cfg.runtime.max_repairs=0
    cfg.agents['molmonitor'].enabled=False; cfg.runtime.trace_dir=Path('outputs/test_forward')
    models=make_models(case); seen={}
    def bio(n,messages):
        return [('inspect_biophysical_context',{'factor':'all'})] if n==1 else [('submit_biology_plan',{'plan':biology})]
    def math_model(n,messages):
        if n==1:
            context=json.loads(messages[1].content); seen.update(context['goal_pools'])
            return [('prepare_function_synthesis',{'direction_id':'repair','query':'flat-bottom local geometry'})]
        if n==2:
            return [('record_function_derivation',{'direction_id':'repair','stage':'source_transfer',
                'record':{'transfer':'Retain the local relation under a conditional chemical hypothesis.', 'unknowns':['Reference applicability unresolved.']}})]
        if n==3:
            return [('construct_direction_potential',{'direction_id':'repair','relations':[relation(case)],'within_direction':'single'})]
        if n==4:
            return [('test_staged_mathematical_design',{k:design[k] for k in ('strategy','conflict_assessment','independent_evaluation','design_audit')})]
        return [('submit_mathematical_design',{})]
    models['molthinker.biology'].respond=bio
    models['molthinker.mathematics'].respond=math_model
    state=AgentRuntime(cfg,models=models).run(packet,report,run_id='unresolved_goal_fixture')
    assert state['status']=='design_deferred',state['trace'][-5:]
    assert seen['compilation_candidate_ids']==[] and seen['unresolved_selected_goal_ids']==['repair']
    assert state['mathematical_design']['directions'][0]['status']=='design_only'
    assert state['reward_design_deferral']['blocked_directions']==['repair']
    assert models['molexecutor'].calls==0
    assert not any(e['kind']=='error' for e in state['trace'])
    source=state['expert_workspace_history'][0]['source_transfers'][0]
    assert source['retrievals'] and source['scope'].startswith('Actual sources')
    assert state['expert_workspace_history'][0]['decisions']['mathematics']['repair']['source_transfer']['unknowns']


def test_unselected_context_goal_can_be_researched_without_becoming_objective(case):
    packet,report,biology,design,_=case
    ident=next(d['direction_id'] for d in biology['directions'] if d['direction_id']!='repair')
    cfg=load_config(); cfg.thinker.require_design_audit=False; cfg.thinker.external_research=False
    cfg.agents['molmonitor'].enabled=False; cfg.runtime.trace_dir=Path('outputs/test_forward')
    models=make_models(case)
    def math_model(n,messages):
        if n==1:
            return [('prepare_function_synthesis',{'direction_id':ident,'function_ids':['P06']}),
                    ('search_direction_knowledge',{'direction_id':'repair','query':'flat-bottom local_geometry'})]
        if n==2:
            submitted=copy.deepcopy(design)
            submitted['directions'][0]['retrieval_ids']=[observations(messages,'search_direction_knowledge')[0]['retrieval_id']]
            return [('test_mathematical_design',{'design':submitted})]
        return [('submit_mathematical_design',{})]
    models['molthinker.mathematics'].respond=math_model
    state=AgentRuntime(cfg,models=models).run(packet,report,run_id='context_research_fixture')
    assert state['status']=='validated',state['trace'][-5:]
    assert [d['direction_id'] for d in state['mathematical_design']['directions']]==['repair']
    assert any(r['direction_id']==ident for r in state['retrieval_records'])
