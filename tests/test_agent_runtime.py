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
    return cfg


def test_offline_graph_and_input_immutability(config, inputs):
    packet, report = inputs
    original = copy.deepcopy(packet)
    state = AgentRuntime(config).run(packet, report)
    assert state['status'] == 'validated'
    assert packet == original
    assert state['validation']['passed']
    assert Path(state['checkpoint_path']).is_file()
    assert {e['node'] for e in state['trace']} == {'reader','thinker','executor','monitor'}


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


def test_adapter_receives_retunes_before_replan(config, inputs):
    config.monitoring.warmup = 2
    config.monitoring.persistence = 1
    config.monitoring.cooldown = 0
    config.runtime.max_replans = 1
    config.runtime.max_segments = 8
    requests = []
    def adapter(**kwargs):
        requests.append(kwargs['request'])
        return {'done':False,'metrics':{'movement':1. if len(requests)<=2 else 20.}}
    state = AgentRuntime(config,inference_adapter=adapter,approve_inference=True).run(*inputs,execute=True)
    assert state['status'] == 'replan_limit'
    assert [r['strength'] for r in requests[:5]] == [1.,1.,1.,.5,.25]
    actions = [e['summary'] for e in state['trace'] if e['node']=='monitor']
    assert actions.index('Monitor retune_strength') < actions.index('Monitor revise_reward')


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


def test_all_four_api_agents_use_actual_tool_calls(config, inputs):
    config.mode = 'api'
    base = derive(*inputs, ROOT/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md')
    models = {
        'molreader':ScriptModel([(n,{}) for n in ['inspect_geometry','inspect_chemistry','inspect_uncertainty','submit_diagnosis']]),
        'molthinker':ScriptModel([('derive_reward_candidates',{}),('search_reviewed_reward_knowledge',{'query':'geometry'}),('submit_reward_plan',{
            'term_ids':[t['term_id'] for t in base['terms']], 'weights':[t['weight'] for t in base['terms']], 'scales':[t['scale'] for t in base['terms']]})]),
        'molexecutor':ScriptModel([('inspect_reward_program',{}),('test_reward_program',{}),('submit_tested_program',{})]),
        'molmonitor':ScriptModel([('inspect_monitor_event',{}),('acknowledge_monitor_route',{})]),
    }
    state = AgentRuntime(config,models=models).run(*inputs)
    assert state['status'] == 'validated', state['errors']
    assert all(m.invocations == 1 for m in models.values())
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
                assert messages[2].additional_kwargs['reasoning_details'] == details
            return AIMessage(content='', additional_kwargs={'reasoning_details': details},
                             tool_calls=[{'name': 'check', 'args': {'step': self.turn},
                                          'id': f'call_{self.turn}'}])

    state = {'run_id': 'reasoning_test'}
    run_tools(TwoTurnModel(), [CheckTool()], instructions='Check two steps', context={},
              state=state, node='reader', max_steps=2, completed=lambda: completed['value'])
    assert 'test-only private detail' not in json.dumps(state)
