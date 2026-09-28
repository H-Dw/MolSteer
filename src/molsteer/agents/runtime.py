"""Four independent API-default agents driven by a bounded LangGraph workflow."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import uuid
from langchain_core.tools import tool
from .config import AGENT_NAMES, AgentSystemConfig, SecretStore, load_config
from .models import create_chat_model
from .state import initial_state, validate_packet, validate_report
from .trace import append_trace, save_checkpoint, save_trace
from .loop import run_tools
from .reader import reader_tools
from .thinker import thinker_tools
from .executor import executor_tools, validate_and_test_reward
from .monitor import RobustMonitor, monitor_metrics


class AgentRuntime:
    """API mode never falls back. Offline mode explicitly runs domain algorithms.

    ``inference_adapter`` is a host-approved callable invoked with keyword args
    packet, reward_spec, execution_result (validation), request. It must return
    {done: bool, metrics: dict[str,float], ...}. It owns sampler state/masks and
    consumes request.strength. No live generator is bundled with this workflow.
    """
    def __init__(self, config: AgentSystemConfig | None=None, *, models=None,
                 knowledge_path=None, search_fn=None, compute_fn=None,
                 inference_adapter=None, approve_inference=False):
        self.config=config or load_config()
        self.models=dict(models or {})
        if set(self.models)-set(AGENT_NAMES): raise ValueError('unknown injected model agent')
        self.knowledge_path=Path(knowledge_path or self.config.repo_root/'knowledge'/'Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md').resolve()
        self.knowledge_path.relative_to((self.config.repo_root/'knowledge').resolve())
        self.search_fn=search_fn; self.compute_fn=compute_fn
        self.inference_adapter=inference_adapter; self.approve_inference=approve_inference
        self.monitor=None

    def _model(self,name):
        if name not in self.models: self.models[name]=create_chat_model(self.config,name,SecretStore(self.config))
        return self.models[name]

    def _loop(self,state,node,tools,context,instructions,completed):
        run_tools(self._model('mol'+node),tools,instructions=instructions,context=context,state=state,node=node,
                  max_steps=self.config.runtime.max_agent_steps,max_repairs=self.config.runtime.max_repairs,completed=completed)

    def reader(self,state):
        tools,result=reader_tools(state['packet'],state.get('diagnostic_report') or None)
        if self.config.mode=='offline':
            for t in tools:
                if t.name!='read_bound_statepack':
                    observation=t.invoke({})
                    append_trace(state,node='reader',kind='tool',summary='Offline evidence path',tool_name=t.name,output=observation)
        else:
            self._loop(state,'reader',tools,{'packet_id':state['packet']['packet_id']},
                       'Inspect geometry, chemistry and uncertainty via their separate tools, then submit_diagnosis.',lambda:'report' in result)
        state['diagnostic_report']=result['report']; state['route']='thinker'
        return state

    def thinker(self,state):
        creative = self.config.skill_path.parent.name!='molthinker-reward-selection'
        tools,result,base=thinker_tools(state['packet'],state['diagnostic_report'],self.knowledge_path,
                                       search_fn=self.search_fn,compute_fn=self.compute_fn,feedback=state.get('monitor_event'),
                                       mode='creativity' if creative else 'selection')
        if self.config.mode=='offline':
            result['spec']=base
            append_trace(state,node='thinker',kind='tool',summary='Explicit offline evidence-bound derivation',tool_name='derive_reward_candidates',output=base)
        else:
            instructions = ('First record_task_plan with concise steps and evidence IDs. Inspect candidate measurements and '
                'retrieve reviewed function entries. Discover the smallest sufficient core repair targets, distinguish '
                'proxies from causes, and submit_reward_design with target groups, physical normalizations, omitted '
                'candidate dispositions, a justified declarative objective tree and a rejected alternative. '
                'If the correct shape, prerequisites or constraints cannot be represented by the safe backend, '
                'call defer_reward_design with the missing requirements; never substitute a flat weight list. '
                'Do not copy any previous balance formula. No raw chain of thought.'
                if creative else
                'First record_task_plan with concise steps and evidence IDs. Use derive_reward_candidates and reviewed '
                'retrieval, then submit_reward_plan with candidate term IDs, weights and scales. No raw chain of thought.')
            self._loop(state,'thinker',tools,{'report':state['diagnostic_report'],'feedback':state.get('monitor_event',{})},
                       state['skill_text']+'\n'+instructions,lambda:'spec' in result or 'deferral' in result)
        if 'deferral' in result:
            state['reward_design_deferral']=result['deferral']
            state['status']='design_deferred';state['route']='done'
            append_trace(state,node='thinker',kind='decision',summary='Creative reward design deferred; no template compiled',
                         output=result['deferral'])
            return state
        state['reward_spec']=result['spec']
        state['plan']={'kind':'ControlPlan','reward_id':result['spec']['reward_id'],
                       'task_plan':result.get('task_plan',{'status':'not_provided'}),
                       'objective_terms':[{'term_id':t['term_id'],'evidence_id':t['reference_evidence_id'],'weight':t['weight'],'scale':t['scale']} for t in result['spec']['terms']],
                       'constraints':['fixed input packet','frozen graph tests only','host adapter required for inference'],
                       'continuation':{'strength':state['strength'],'max_segments':self.config.runtime.max_segments},
                       'retrieval':result['spec']['retrieval'],'checks':result['spec']['required_validation'],
                       'reward_design':result['spec'].get('design'),
                       'summary':('Selected evidence-bound core targets and a declarative objective architecture'
                                  if result['spec'].get('design') else
                                  'Selected supported evidence-bound coordinate penalties')+
                                 '; live control remains adapter-dependent.'}
        state['validation']={}; state['route']='executor'
        append_trace(state,node='thinker',kind='decision',summary=state['plan']['summary'],output=state['plan'])
        return state

    def executor(self,state):
        # Compile/test once per plan or strength change, not on every live segment.
        key=(state['reward_spec']['reward_id'],state['strength'])
        if state.get('validation_key')!=list(key):
            tools,result=executor_tools(state['packet'],state['reward_spec'],state['diagnostic_report'],strength=state['strength'])
            if self.config.mode=='offline':
                result['validation']=validate_and_test_reward(state['packet'],state['reward_spec'],state['diagnostic_report'],strength=state['strength'])
                append_trace(state,node='executor',kind='tool',summary='Validated numerical reward program',tool_name='test_reward_program',output=result['validation'])
            else:
                self._loop(state,'executor',tools,{'reward_id':key[0],'strength':key[1]},
                           'Inspect the declarative reward program; test_reward_program and repair invalid bounded test settings if needed. submit_tested_program only after success. No arbitrary code execution.',lambda:'validation' in result)
            state['validation']=result['validation']; state['validation_key']=list(key)
        if state['execute']:
            if not self.approve_inference or not callable(self.inference_adapter): raise RuntimeError('approved inference adapter required')
            request={'segment':state['segments'],'strength':state['strength'],'run_id':state['run_id'],
                     'monitor_event':deepcopy(state.get('monitor_event',{}))}
            result=self.inference_adapter(packet=deepcopy(state['packet']),reward_spec=deepcopy(state['reward_spec']),execution_result=deepcopy(state['validation']),request=request)
            if not isinstance(result,dict) or type(result.get('done')) is not bool or not isinstance(result.get('metrics'),dict): raise ValueError('adapter must return done boolean and metrics object')
            state['execution_result']=deepcopy(result)
        else:
            trials=state['validation']['trials']
            state['execution_result']={'done':True,'mode':'validation_only','metrics':{'penalty_after':sum(t['penalty_after'] for t in trials)},'validation':state['validation']}
        state['segments']+=1; state['route']='monitor'
        append_trace(state,node='executor',kind='observation',summary='Live adapter segment' if state['execute'] else 'Numerical tests only; inference not run',output=state['execution_result'])
        return state

    def monitoring(self,state):
        event=self.monitor.observe(monitor_metrics(state['execution_result']),step=state['segments'])
        if self.config.mode=='api':
            reviewed={}
            @tool
            def inspect_monitor_event() -> dict:
                """Read per-metric robust statistics and the non-overridable host route."""
                reviewed['read']=True
                return deepcopy(event)
            @tool
            def acknowledge_monitor_route() -> dict:
                """Acknowledge the deterministic safety route after inspecting metrics; cannot override gates."""
                if not reviewed.get('read'): raise ValueError('inspect monitor event first')
                reviewed['done']=True
                return {'status':'accepted','route':event['route'],'action':event['action']}
            self._loop(state,'monitor',[inspect_monitor_event,acknowledge_monitor_route],{'segment':state['segments']},
                       'Inspect monitor event then acknowledge the host route. Single fluctuations are held; sustained shifts retune Executor before Thinker revision; never bypass hard safety stops.',lambda:reviewed.get('done',False))
        state['monitor_event']=event
        state['strength']=event.get('strength',state['strength'])
        if event['route']=='stop': state['status']='safety_stopped'; state['route']='done'
        elif event['route']=='thinker':
            if state['replans']>=self.config.runtime.max_replans: state['status']='replan_limit'; state['route']='done'
            else: state['replans']+=1; state['route']='thinker'
        elif event['route']=='executor': state['route']='executor'
        elif state['execution_result']['done']:
            state['status']='completed' if state['execute'] else 'validated'; state['route']='done'
        else: state['route']='executor'
        if state['route']!='done' and state['segments']>=self.config.runtime.max_segments:
            state['status']='segment_limit'; state['route']='done'
        append_trace(state,node='monitor',kind='decision',summary='Monitor '+event['action'],output=event)
        return state

    def run(self,packet,diagnostic_report=None,run_id=None,execute=False):
        validate_packet(packet)
        if diagnostic_report is not None: validate_report(diagnostic_report,packet)
        run_id=run_id or 'run_'+uuid.uuid4().hex[:24]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',run_id): raise ValueError('invalid run id')
        if type(execute) is not bool: raise ValueError('execute must be boolean')
        if execute and (not self.approve_inference or not callable(self.inference_adapter)): raise ValueError('execute requires an explicitly approved inference adapter')
        skill=self.config.skill_path.read_text(encoding='utf-8')
        if self.config.skill_path.parent.name=='molthinker-reward-creativity':
            reference=self.config.skill_path.parent/'references'/'core-target-and-shape.md'
            skill+='\n\n# Required core-target and mathematical-shape reference\n'+reference.read_text(encoding='utf-8')
        state=initial_state(run_id=run_id,packet=deepcopy(packet),diagnostic_report=deepcopy(diagnostic_report))
        state.update(execute=execute,segments=0,strength=self.config.monitoring.max_strength,skill_text=skill,
                     config=self.config.model_dump(mode='json'),skill_sha256=hashlib.sha256(skill.encode()).hexdigest(),
                     config_sha256=hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),plan={},validation={},validation_key=[])
        self.monitor=RobustMonitor(self.config.monitoring)
        try:
            if self.config.mode=='api':
                for name in AGENT_NAMES: self._model(name)
            from .workflow import build_workflow
            state=build_workflow(self).invoke(state,{'recursion_limit':4*self.config.runtime.max_segments+4*self.config.runtime.max_replans+10})
        except Exception as exc:
            state['status']='failed'; state['route']='error'; state['errors']=[type(exc).__name__+': workflow failed; review validated tool observations']
            append_trace(state,node='runtime',kind='error',summary='Workflow failed without fallback',output={'error_type':type(exc).__name__})
        state.pop('skill_text',None)
        state['trace_path']=str(save_trace(state,self.config.trace_dir))
        state['checkpoint_path']=str(save_checkpoint(state,self.config.trace_dir))
        return state


def run_agent_workflow(packet,diagnostic_report=None,*,config=None,run_id=None,execute=False,**kwargs):
    return AgentRuntime(config,**kwargs).run(packet,diagnostic_report,run_id,execute)

__all__=['AgentRuntime','run_agent_workflow']
