import copy
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from molsteer.agents.config import load_config
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.loop import run_tools
from molsteer.agents.monitor import RobustMonitor
from molsteer.molthinker.planner import derive
from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/5i0b_A__5vef_M77/ligand_002/t_0.50'

@pytest.fixture
def inputs():
    return tuple(json.loads((EXAMPLE / name).read_text(encoding='utf-8')) for name in ['StatePacket.json','DiagnosticReport.json'])

@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.runtime.trace_dir = Path('outputs/test_agent_runtime')
    cfg.mode = 'offline'
    cfg.thinker.architecture = 'single'  # Explicit legacy compatibility coverage.
    return cfg


def test_offline_graph_and_input_immutability(config, inputs):
    packet, report = inputs
    original = copy.deepcopy(packet)
    state = AgentRuntime(config).run(packet, report)
    assert state['status'] == 'validated'
    assert packet == original
    assert state['validation']['passed']
    assert Path(state['checkpoint_path']).is_file()
    assert {e['node'] for e in state['trace']} == {'reader','thinker','executor'}


def test_disabled_monitor_is_skipped_and_validation_finishes(config,inputs,monkeypatch):
    config.agents['molmonitor'].enabled=False
    runtime=AgentRuntime(config)
    monkeypatch.setattr(runtime,'monitoring',lambda *a:pytest.fail('Disabled monitor was invoked'))
    state=runtime.run(*inputs)
    assert state['status']=='validated' and state['validation']['passed']
    assert runtime.monitor is None
    assert {e['node'] for e in state['trace']}=={'reader','thinker','executor'}


def test_direct_execution_requests_complete_native_suffix_once(config,inputs):
    config.agents['molmonitor'].enabled=False
    requests=[]
    def adapter(**kwargs):
        requests.append(kwargs['request'])
        return {'done':True,'metrics':{'movement':0.2}}
    state=AgentRuntime(config,inference_adapter=adapter,approve_inference=True).run(*inputs,execute=True)
    assert state['status']=='completed' and len(requests)==1
    assert requests[0]['guidance_weight']==1. and requests[0]['continuation']=='all_remaining_native_steps'
    assert not any(e['node']=='monitor' for e in state['trace'])


def test_monitor_persistence_and_escalation():
    monitor = RobustMonitor(warmup=2, window=4, persistence=2, cooldown=0, max_retunes=2)
    for _ in range(2): monitor.observe({'energy':1000., 'displacement':.1})
    events = [monitor.observe({'energy':1000.,'displacement':10.}) for _ in range(6)]
    assert events[0]['route'] == 'stable'
    assert [e['route'] for e in events[1::2]] == ['executor','executor','thinker']
    assert events[1]['strength'] == .5
    assert events[3]['strength'] == .25
    assert events[-1]['scores']['energy']['score'] == 0
    assert monitor.observe({'x':float('nan')})['route'] == 'stop'


def test_direct_execution_does_not_retune_or_replan_even_if_legacy_monitor_enabled(config, inputs):
    config.agents['molmonitor'].enabled = True
    config.monitoring.max_strength = .1
    config.runtime.guidance_weight = 100.
    config.runtime.max_segments = 1
    requests = []
    def adapter(**kwargs):
        requests.append(kwargs['request'])
        return {'done': True, 'metrics': {'movement': 20.}}
    state = AgentRuntime(config, inference_adapter=adapter, approve_inference=True).run(*inputs, execute=True)
    assert state['status'] == 'completed' and len(requests) == 1
    assert requests[0]['guidance_weight'] == 100.
    assert not any(e['node'] == 'monitor' for e in state['trace'])


def test_incomplete_adapter_result_is_failure_without_silent_segment_loop(config, inputs):
    state = AgentRuntime(config, inference_adapter=lambda **k: {'done':False, 'metrics':{}},
                         approve_inference=True).run(*inputs, execute=True)
    assert state['status'] == 'failed'


def test_inference_requires_host_approval(config, inputs):
    with pytest.raises(ValueError,match='approved'):
        AgentRuntime(config).run(*inputs,execute=True)


class ScriptModel:
    def __init__(self, calls):
        self.calls = calls
        self.invocations = 0
    def bind_tools(self, tools):
        self.names = {t.name for t in tools}
        return self
    def invoke(self, messages):
        self.invocations += 1
        return AIMessage(content='PRIVATE_PROVIDER_REASONING_MUST_NOT_PERSIST',tool_calls=[
            {'name':name,'args':args,'id':f'call_{self.invocations}_{i}'} for i,(name,args) in enumerate(self.calls)])


def test_reader_thinker_api_and_deterministic_executor(config, inputs):
    config.mode = 'api'
    base = derive(*inputs, ROOT/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md')
    term_ids=[base['terms'][i]['term_id'] for i in (0,2)]
    design={
        'target_groups':[{'role':'repair','term_ids':term_ids,
                          'repair_predicate':'The selected local distances satisfy their evidenced intervals.',
                          'rationale':'This synthetic API test exercises an explicitly chosen core target set.',
                          'falsifier':'Independent geometry screening can remain abnormal despite objective descent.'}],
        'normalizations':[{'term_id':t['term_id'],'scale':t['scale'],
                           'origin':'Reported screening scale retained for this bounded API test.'}
                          for t in base['terms'] if t['term_id'] in term_ids],
        'omitted':[{'term_id':t['term_id'],'role':'monitor',
                    'reason':'This synthetic test leaves the other diagnostic candidate under monitoring.'}
                   for t in base['terms'] if t['term_id'] not in term_ids],
        'objective_tree':{'op':'lp_norm','p':2,'children':[{'op':'term','term_id':t} for t in term_ids]},
        'architecture_reason':'An L2 violation norm requires simultaneous progress on the selected local deficits.',
        'rejected_alternatives':['A flat sum can hide a persistent local violation behind several smaller gains.'],
        'mathematical_audit':{
            'zero_set':'The tree is zero only when both selected nonnegative residuals are zero.',
            'marginal_sensitivity':'Each active residual receives pressure proportional to its own magnitude.',
            'constraint_and_gradient_path':'Copy-coordinate derivatives are checked; the live generator Jacobian remains untested.',
            'failure_mode':'A surrogate residual can fall while independent final geometry remains abnormal.'},
        'evaluation_plan':{
            'independent_measurement':'Recheck localized geometry and chemical validity after a matched continuation.',
            'matched_native_status':'A matched native continuation is not supplied in this synthetic API test.'},
    }
    models = {
        'molreader':ScriptModel([(n,{}) for n in ['inspect_geometry','inspect_chemistry','inspect_uncertainty','submit_diagnosis']]),
        'molthinker':ScriptModel([('derive_reward_candidates',{}),('search_reviewed_reward_knowledge',{'query':'geometry'}),
                                  ('submit_reward_design',{'term_ids':term_ids,'design':design})]),
        'molexecutor':ScriptModel([('inspect_reward_program',{}),('test_reward_program',{}),('submit_tested_program',{})]),
        'molmonitor':ScriptModel([('inspect_monitor_event',{}),('acknowledge_monitor_route',{})]),
    }
    state = AgentRuntime(config,models=models).run(*inputs)
    assert state['status'] == 'validated', state['errors']
    assert state['reward_spec']['design']['objective_tree']['op']=='lp_norm'
    assert len(state['reward_spec']['terms'])==2
    guard=json.loads((ROOT/'experiments/guidance/creativity.json').read_text(encoding='utf-8'))
    program,_,editable,_=compile_validated_agent_checkpoint(state['checkpoint_path'],guard)
    assert program['mode']=='agent_design'
    assert program['design']['objective_tree']['op']=='lp_norm'
    assert editable==sorted({a for t in state['reward_spec']['terms'] for a in t['atom_ids']})
    assert models['molreader'].invocations == models['molthinker'].invocations == 1
    assert models['molexecutor'].invocations == models['molmonitor'].invocations == 0
    assert 'PRIVATE_PROVIDER_REASONING' not in json.dumps(state)
    assert 'PRIVATE_PROVIDER_REASONING' not in Path(state['trace_path']).read_text(encoding='utf-8')


def test_model_failure_does_not_fall_back(config, inputs):
    config.mode='api'
    class Broken(ScriptModel):
        def invoke(self,messages): raise RuntimeError('credential=must-not-leak')
    models={name:Broken([]) for name in config.agents}
    state=AgentRuntime(config,models=models).run(*inputs)
    assert state['status']=='failed'
    assert 'must-not-leak' not in json.dumps(state)


def test_creative_agent_can_defer_unsupported_shape_without_template(config, inputs):
    config.mode='api'
    _,report=inputs
    evidence_id=next(iter(report['evidence_index']))
    models={
        'molreader':ScriptModel([(n,{}) for n in ['inspect_geometry','inspect_chemistry','inspect_uncertainty','submit_diagnosis']]),
        'molthinker':ScriptModel([('defer_reward_design',{
            'core_target_evidence_ids':[evidence_id],
            'reason':'The core defect requires a coupled physical energy unavailable in this backend.',
            'missing_requirements':['Validated graph-dependent force-field parameters'],
            'proposed_shape':'A graph-conditioned local strain energy with explicit preservation constraints.'})]),
        'molexecutor':ScriptModel([]),'molmonitor':ScriptModel([]),
    }
    state=AgentRuntime(config,models=models).run(*inputs)
    assert state['status']=='design_deferred'
    assert state['reward_spec']=={}
    assert state['reward_design_deferral']['status']=='design_only'
    assert models['molexecutor'].invocations==models['molmonitor'].invocations==0


def test_tool_loop_keeps_openrouter_reasoning_details_only_in_memory():
    details = [{'type': 'reasoning.text', 'text': 'test-only private detail'}]
    completed = {'value': False}

    class CheckTool:
        name = 'check'

        def invoke(self, args):
            completed['value'] = args['step'] == 2
            return {'status': 'ok'}

    class TwoTurnModel:
        def __init__(self):
            self.turn = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.turn += 1
            if self.turn == 2:
                previous=next(message for message in messages if isinstance(message,AIMessage))
                assert previous.additional_kwargs['reasoning_details'] == details
            return AIMessage(content='', additional_kwargs={'reasoning_details': details},
                             tool_calls=[{'name': 'check', 'args': {'step': self.turn},
                                          'id': f'call_{self.turn}'}])

    state = {'run_id': 'reasoning_test'}
    run_tools(TwoTurnModel(), [CheckTool()], instructions='Check two steps', context={},
              state=state, node='reader', max_steps=2, completed=lambda: completed['value'])
    assert 'test-only private detail' not in json.dumps(state)
