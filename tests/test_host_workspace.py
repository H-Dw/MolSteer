from copy import deepcopy
import json
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from molsteer.agents.host_workspace import HostWorkspace, NO_PROGRESS
from molsteer.agents.loop import run_tools
from molsteer.agents.host_workspace import record_usage


def test_reference_versions_canonical_aliases_and_wording_do_not_count_as_progress():
    host = HostWorkspace({}, 'math')
    direction = dict(direction_id='a', expression={'op':'constant', 'value':1}, formula='one')
    host.save('a', direction)
    host.observe('construct', {}, {'status':'draft_only'})
    design = dict(directions=[direction], strategy={'mode':'scalar_potential'})
    host.save('design', design)
    host.observe('construct', {}, {'status':'draft_only'})
    assert 'content' not in host.data['artifacts']['a']
    assert host.read('a')['value'] == direction
    for i in range(3):
        changed = dict(direction, formula='wording '+str(i))
        host.save('a', changed)
        host.observe('patch_candidate', {}, {'status':'draft_only'})
    assert host.summary()['no_progress_notice'] == NO_PROGRESS
    assert host.read('design', force_full=True)['value']['directions'][0]['formula'] == 'wording 2'
    changed['expression']['value'] = 2
    host.save('a', changed)
    host.observe('patch_candidate', {}, {'status':'draft_only'})
    assert host.summary()['no_progress_notice'] is None
    # Returning to an earlier computational candidate is not new progress.
    changed['expression']['value'] = 1
    host.save('a', changed)
    host.observe('patch_candidate', {}, {'status':'draft_only'})
    assert host.data['idle_actions'] == 1


def test_contract_reads_are_paged_and_repeat_is_receipt():
    host = HostWorkspace({}, 'math')
    host.save('contract', {'schema':{'long':['content']}, 'units':['angstrom']}, kind='contract')
    first = host.read('contract', ['units'])
    assert first['value'] == ['angstrom']
    assert 'value' not in host.read('contract', ['units'])
    assert host.read('contract', ['schema'])['value'] == {'long':['content']}
    assert host.read('contract', ['units'], True)['value'] == first['value']


def test_routing_keeps_legacy_calls_and_notice_uses_normal_requests_only():
    state = {'trace':[]}
    host = HostWorkspace(state, 'thinker.mathematics', threshold=3)
    done = {}
    @tool
    def inspect_fact() -> dict:
        """Read a deterministic existing fact."""
        return {'status':'available', 'fact':1}
    @tool
    def test_legacy_full_design() -> dict:
        """Known compatibility tool hidden in the default research phase."""
        done['yes'] = True
        return {'status':'tested', 'passed':True}
    class Model:
        def __init__(self):
            self.calls = 0
            self.inputs = []
            self.shown = []
        def bind_tools(self, tools):
            self.shown.append([t.name for t in tools])
            return self
        def invoke(self, messages):
            self.inputs.append(deepcopy(messages))
            name = 'inspect_fact' if self.calls < 4 else 'test_legacy_full_design'
            self.calls += 1
            return AIMessage(content='', tool_calls=[dict(id=str(self.calls), name=name, args={})])
    model = Model()
    run_tools(model, [inspect_fact, test_legacy_full_design], instructions='test', context={}, state=state,
        node='thinker.mathematics', max_steps=8, completed=lambda:bool(done), host_workspace=host)
    assert model.calls == 5
    assert all('test_legacy_full_design' not in tools for tools in model.shown)
    assert NO_PROGRESS in model.inputs[-1][-1].content
    assert len(state['api_usage']) == 5
    assert all(r['usage_status'] == 'unknown' and r['usage'] is None for r in state['api_usage'])
    assert all(r['request_bytes'] > 0 and r['tool_definition_bytes'] > 0 for r in state['api_usage'])
    assert not any(e['kind'] == 'error' for e in state['trace'])


def test_routing_can_reopen_research_without_discarding_candidate():
    host = HostWorkspace({}, 'math')
    host.save('a', {'expression':{'op':'constant', 'value':1}})
    assert host.phase()[0] == 'candidate'
    tools = {t.name:t for t in host.tools()}
    tools['resolve_evidence_question'].invoke(dict(question_id='q', question='Missing local contact measurement'))
    assert host.phase()[0] == 'research'
    assert host.read('a')['value']['expression']['value'] == 1
    tools['resolve_evidence_question'].invoke(dict(question_id='q', question='Missing local contact measurement', deferred=True))
    assert host.phase()[0] == 'candidate'


def test_usage_preserves_provider_reported_tokens_and_cost_without_new_request():
    from langchain_core.messages import HumanMessage
    state = {}
    receipt = dict(id='fixture',model='provider-model',usage={'prompt_tokens':19,'completion_tokens':7,'cost':.001})
    response = AIMessage(content='', additional_kwargs={'_openrouter_receipt':receipt})
    record_usage(state,'thinker.mathematics','candidate',response,[HumanMessage(content='test')],[])
    assert state['api_usage'][0]['usage'] == receipt['usage']
    assert state['api_usage'][0]['usage_status'] == 'reported'
