"""Reader and Thinker agents followed by deterministic scalar execution."""
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
from .workflow_guidance import load_workflow_materials
from .raw_reference_tools import raw_reference_tools
from molsteer.molreader.raw_reference import load_raw_reference, raw_reference_summary, bound_raw_context


class AgentRuntime:
    """API mode never falls back. Offline mode explicitly runs domain algorithms.

    ``inference_adapter`` is a host-approved callable invoked with keyword args
    packet, reward_spec, execution_result (validation), request. It must return
    {done: True, metrics: dict[str,float], ...} after the full native suffix.
    It owns sampler state/masks and consumes request.guidance_weight exactly once.
    """
    def __init__(self, config: AgentSystemConfig | None=None, *, models=None,
                 knowledge_path=None, search_fn=None, compute_fn=None,
                 inference_adapter=None, approve_inference=False, model_dynamics=None,
                 fetch_fn=None, research_providers=None):
        self.config=config or load_config()
        self.models=dict(models or {})
        allowed=set(AGENT_NAMES)|{'molthinker.'+r for r in ('biology','mathematics','researcher')}
        if set(self.models)-allowed: raise ValueError('unknown injected model agent')
        self.knowledge_path=Path(knowledge_path or self.config.repo_root/'knowledge'/'Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md').resolve()
        self.knowledge_path.relative_to((self.config.repo_root/'knowledge').resolve())
        self.search_fn=search_fn; self.compute_fn=compute_fn
        self.fetch_fn=fetch_fn; self.research_providers=dict(research_providers or {})
        if 'local' in self.research_providers: raise ValueError('The bound local corpus cannot be overridden')
        from .expert_contracts import ModelDynamicsContext
        if model_dynamics is None and callable(getattr(inference_adapter,'describe_dynamics',None)):
            model_dynamics=inference_adapter.describe_dynamics()
        self.model_dynamics=ModelDynamicsContext.model_validate(model_dynamics or {}).model_dump()
        self.inference_adapter=inference_adapter; self.approve_inference=approve_inference
        self.monitor=None

    def _model(self,name):
        if name not in self.models: self.models[name]=create_chat_model(self.config,name,SecretStore(self.config))
        return self.models[name]

    def _loop(self,state,node,tools,context,instructions,completed):
        from .host_workspace import HostWorkspace
        run_tools(self._model('mol'+node),tools,instructions=instructions,context=context,state=state,node=node,
                  max_steps=self.config.runtime.max_agent_steps,max_repairs=self.config.runtime.max_repairs,completed=completed,
                  host_workspace=HostWorkspace(state,node,self.config.runtime.no_progress_actions))

    def reader(self,state):
        raw_context=load_raw_reference(state['packet'],self.config.reader.raw_reference,self.config.repo_root)
        tools,result=reader_tools(state['packet'],state.get('diagnostic_report') or None,
                                  raw_reference_context=raw_context)
        state['raw_reference_context']=raw_context
        from .measurement_supplements import MeasurementSupplements
        self.measurement_supplements=MeasurementSupplements(state['packet'],raw_context,state,self.config.repo_root,
            self.config.trace_dir / (state['run_id']+'_evidence'), self.config.reader.measurement_stage_paths)
        tools.extend(self.measurement_supplements.tools())
        reader_materials=load_workflow_materials(self.config.repo_root/'skills/molreader-diagnose/SKILL.md')
        reader_skill=reader_materials['SKILL.md']['text']
        state['reader_guidance_sources']={k:r['sha256'] for k,r in reader_materials.items()}
        if raw_context['status']!='disabled':
            append_trace(state,node='reader',kind='observation',summary='Saved raw suffix comparison bound to current checkpoint',
                         output=raw_reference_summary(raw_context))
        if self.config.mode=='offline':
            for t in tools:
                if t.name in ('inspect_geometry', 'inspect_chemistry', 'inspect_uncertainty', 'submit_diagnosis'):
                    observation=t.invoke({})
                    append_trace(state,node='reader',kind='tool',summary='Offline evidence path',tool_name=t.name,output=observation)
        else:
            self._loop(state,'reader',tools,{'packet_id':state['packet']['packet_id'],
                       'raw_reference':raw_reference_summary(raw_context)},
                       reader_skill+'\nStart with residual_needs from the raw comparison, then trace current precursors. When configured raw references '
                       'are available, compare selected views, persistent/current and later-emerging risks, sampled repair '
                       'intervals, local contacts/burial/chemical evolution and configured affinity/SA/stability trends. '
                       'Use inspect_raw_goal_trajectory to trace candidate mechanisms across EVERY configured node, '
                       'including current_condition_status, current typed references, recurrence and final diagnosis. '
                       'Use inspect_raw_comparison and read_raw_reference for exact details. Missing coverage is '
                       'unknown; distinguish retyping of an old object from validity of the later chemistry. An observed '
                       'raw final is known even when intervention benefit is unknown. Global improvement does not prove regional contribution. '
                       'Optionally record_raw_reference_analysis with public interpretations and gaps. Keep DiagnosticReport '
                       'bound to the current state, then submit_diagnosis. No extra comparison-call gate.',lambda:'report' in result)
        state['diagnostic_report']=result['report']; state['route']='thinker'
        state['raw_reference_analysis']=result.get('raw_reference_analysis',{})
        if state.get('terminal_graph_review'):
            state['status']='completed';state['route']='done'
        return state

    def thinker(self,state):
        if self.config.thinker.architecture=='dual_expert' and self.config.mode=='api':
            return self._dual_thinker(state)
        creative = self.config.skill_path.parent.name!='molthinker-reward-selection'
        tools,result,base=thinker_tools(state['packet'],state['diagnostic_report'],self.knowledge_path,
                                       search_fn=self.search_fn,compute_fn=self.compute_fn,feedback=state.get('monitor_event'),
                                       mode='creativity' if creative else 'selection')
        raw_context=bound_raw_context(state.get('raw_reference_context'),state['packet'])
        tools.extend(raw_reference_tools(raw_context,state['packet']))
        if self.config.mode=='offline':
            result['spec']=base
            append_trace(state,node='thinker',kind='tool',summary='Explicit offline evidence-bound derivation',tool_name='derive_reward_candidates',output=base)
        else:
            instructions = ('First record_task_plan with concise steps and evidence IDs. Inspect candidate measurements and '
                'retrieve reviewed function entries. Discover the smallest sufficient core repair targets, distinguish '
                'proxies from causes, and submit_reward_design with target groups, physical normalizations, omitted '
                'candidate dispositions, a justified declarative objective tree, a rejected alternative, '
                'mathematical_audit (zero set, sensitivities, gradient path and failure mode), and evaluation_plan. '
                'If the correct shape, prerequisites or constraints cannot be represented by the safe backend, '
                'call defer_reward_design with the missing requirements; never substitute a flat weight list. '
                'Do not copy any previous balance formula. No raw chain of thought.'
                if creative else
                'First record_task_plan with concise steps and evidence IDs. Use derive_reward_candidates and reviewed '
                'retrieval, then submit_reward_plan with candidate term IDs, weights and scales. No raw chain of thought.')
            if state.get('monitor_event',{}).get('kind')=='ChemicalGraphChange':
                instructions += (' Review the previous reward against the fresh graph. Keep native slot IDs; reassign '
                    'chemical roles and support. Retain valid function forms, rederive obsolete bounds, and explain '
                    'changes. A disappeared target is not proof of repair. Do not require the initial graph identity.')
            self._loop(state,'thinker',tools,{'report':state['diagnostic_report'],'feedback':state.get('monitor_event',{}),
                       'raw_reference':raw_reference_summary(raw_context),
                       'raw_reference_analysis':state.get('raw_reference_analysis',{})},
                       state['skill_text']+'\n'+instructions+' Review configured raw comparisons when available: decide '
                       'whether persistent defects, earlier repair or associated region evolution warrant a current goal. '
                       'Raw outcomes are observed context; guided effects remain unknown. Future references cannot bind '
                       'current reward inputs.',lambda:'spec' in result or 'deferral' in result)
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
                       'continuation':{'strength':state['strength'],'guidance_weight':state['strength'],'execution_semantics':'native_scalar_gradient'},
                       'retrieval':result['spec']['retrieval'],'checks':result['spec']['required_validation'],
                       'reward_design':result['spec'].get('design'),
                       'summary':('Selected evidence-bound core targets and a declarative objective architecture'
                                  if result['spec'].get('design') else
                                  'Selected supported evidence-bound coordinate penalties')+
                                 '; live control remains adapter-dependent.'}
        state['validation']={}; state['route']='executor'
        append_trace(state,node='thinker',kind='decision',summary=state['plan']['summary'],output=state['plan'])
        return state

    def _dual_thinker(self,state):
        from .experts import run_experts
        if state.get('replans',0):
            latest=state.get('execution_result',{}).get('packet')
            if latest is None:
                state.update(status='design_deferred',route='done',reward_spec={},
                             reward_design_deferral={'status':'design_only','reason':'Live replan requires a fresh StatePacket'})
                return state
            validate_packet(latest)
            from molreader.localized_report import make_localized_report
            report=(state.get('diagnostic_report') if state.get('diagnostic_report',{}).get('packet_id')==latest['packet_id']
                    else state['execution_result'].get('diagnostic_report') or make_localized_report(latest))
            validate_report(report,latest)
            state['packet'],state['diagnostic_report']=deepcopy(latest),deepcopy(report)
        result=run_experts(self,state)
        if 'deferral' in result:
            state.update(reward_spec={},reward_design_deferral=result['deferral'],status='design_deferred',route='done')
            return state
        spec=result['spec']; state['reward_spec']=spec
        state['plan']={'kind':'ControlPlan','reward_id':spec['reward_id'],
                       'biology_plan':spec['biology_plan'],'mathematical_design':spec['mathematical_design'],
                       'retrieval':spec['retrieval'],'checks':spec['required_validation'],
                       'continuation':{'strength':state['strength'],'guidance_weight':state['strength'],'execution_semantics':'native_scalar_gradient'},
                       'summary':'Biology-ranked directions and evidence-derived executable mathematical controls'}
        validation=result.get('validation') or {}
        state.update(validation=validation,
            validation_key=[spec['reward_id'],state['strength']] if validation.get('passed') and validation.get('reward_id')==spec['reward_id'] else [],
            route='executor')
        return state

    def executor(self,state):
        # Deterministic loading/testing; no executor model or tuning round.
        key=(state['reward_spec']['reward_id'],state['strength'])
        if state.get('validation_key')!=list(key):
            validation=validate_and_test_reward(state['packet'],state['reward_spec'],state['diagnostic_report'])
            state['validation']=validation; state['validation_key']=list(key)
            if not validation.get('passed'):
                raise ValueError('Scalar reward numerical validation failed')
            append_trace(state,node='executor',kind='tool',summary='Deterministic scalar reward validation',
                         tool_name='test_reward_program',output=validation)
        if state['execute']:
            if not self.approve_inference or not callable(self.inference_adapter):
                raise RuntimeError('approved inference adapter required')
            request={'guidance_weight':state['strength'], 'strength':state['strength'],
                     'run_id':state['run_id'], 'execution_semantics':'native_scalar_gradient',
                     'continuation':'all_remaining_native_steps'}
            result=self.inference_adapter(packet=deepcopy(state['packet']),reward_spec=deepcopy(state['reward_spec']),
                execution_result=deepcopy(state['validation']),request=request)
            if not isinstance(result,dict) or result.get('done') is not True or not isinstance(result.get('metrics'),dict):
                raise ValueError('Direct adapter must complete the remaining native trajectory or report a calculation failure')
            state['execution_result']=deepcopy(result)
        else:
            trials=state['validation']['trials']
            state['execution_result']={'done':True,'mode':'validation_only',
                'metrics':{'penalty_after':sum(t['penalty_after'] for t in trials)},'validation':state['validation']}
        state['segments']+=1
        state['status']='completed' if state['execute'] else 'validated'
        state['route']='done'
        append_trace(state,node='executor',kind='observation',summary='Full native continuation' if state['execute'] else
            'Numerical tests only; inference not run',output=state['execution_result'])
        return state

    def monitoring(self,state):
        event=self.monitor.observe(monitor_metrics(state['execution_result']),step=state['segments'])
        latest=state['execution_result'].get('packet')
        if self.config.monitoring.graph_review_enabled and latest is not None and event['route']!='stop':
            validate_packet(latest)
            from molsteer.molmonitor.graph_review.changes import packet_change_event
            changed=packet_change_event(state['packet'],latest,step=state['segments'],
                                       parent_program_id=state['reward_spec'].get('reward_id'))
            if changed:
                event={**changed,'strength':state['strength'],'numerical_monitor':event}
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
                       'Inspect monitor event then acknowledge the host route. Chemical graph changes activate MolReader before MolThinker; numerical fluctuations use retuning. Never bypass hard safety stops.',lambda:reviewed.get('done',False))
        state['monitor_event']=event
        state['strength']=event.get('strength',state['strength'])
        if event['route']=='stop': state['status']='safety_stopped'; state['route']='done'
        elif event['route']=='reader':
            if state['execution_result']['done']:
                state['terminal_graph_review']=deepcopy(event)
                state['packet']=deepcopy(latest);state['diagnostic_report']={}
                state['route']='reader'
            elif state['replans']>=self.config.runtime.max_replans:
                state['status']='replan_limit';state['route']='done'
            else:
                state['replans']+=1
                state['packet']=deepcopy(latest)
                # Force a fresh reading; an adapter's previous diagnosis must not
                # bypass the Reader when chemical roles have changed.
                state['diagnostic_report']={}
                state['validation']={};state['validation_key']=[]
                state['route']='reader'
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

    def run(self,packet,diagnostic_report=None,run_id=None,execute=False,feedback=None):
        validate_packet(packet)
        if diagnostic_report is not None: validate_report(diagnostic_report,packet)
        run_id=run_id or 'run_'+uuid.uuid4().hex[:24]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',run_id): raise ValueError('invalid run id')
        if type(execute) is not bool: raise ValueError('execute must be boolean')
        if execute and (not self.approve_inference or not callable(self.inference_adapter)): raise ValueError('execute requires an explicitly approved inference adapter')
        skill=self.config.skill_path.read_text(encoding='utf-8')
        if self.config.thinker.architecture=='single' and self.config.skill_path.parent.name=='molthinker-reward-creativity':
            reference=self.config.skill_path.parent/'references'/'core-target-and-shape.md'
            skill+='\n\n# Required core-target and mathematical-shape reference\n'+reference.read_text(encoding='utf-8')
        state=initial_state(run_id=run_id,packet=deepcopy(packet),diagnostic_report=deepcopy(diagnostic_report))
        state.update(execute=execute,segments=0,strength=self.config.runtime.guidance_weight,skill_text=skill,
                     config=self.config.model_dump(mode='json'),skill_sha256=hashlib.sha256(skill.encode()).hexdigest(),
                     config_sha256=hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),plan={},validation={},validation_key=[])
        state['model_dynamics']=deepcopy(self.model_dynamics)
        state['model_dynamics']['execution_scope']=self.config.thinker.execution_scope
        if self.config.thinker.architecture=='dual_expert':
            state['workflow_materials']=load_workflow_materials(self.config.skill_path)
            state['workflow_reference_digests']={ident:row['sha256'] for ident,row in state['workflow_materials'].items()}
        if feedback is not None:state['monitor_event']=deepcopy(feedback)
        self.monitor=None  # Independent opt-in monitor is not part of direct execution.
        try:
            if self.config.mode=='api':
                for name in AGENT_NAMES:
                    if not self.config.agents[name].enabled or name in ('molexecutor','molmonitor'):continue
                    if name!='molthinker' or self.config.thinker.architecture=='single': self._model(name)
            from .workflow import build_workflow
            state=build_workflow(self).invoke(state,{'recursion_limit':4*self.config.runtime.max_segments+4*self.config.runtime.max_replans+10})
        except Exception as exc:
            state['status']='failed'; state['route']='error'; state['errors']=[type(exc).__name__+': workflow failed; review validated tool observations']
            append_trace(state,node='runtime',kind='error',summary='Workflow failed without fallback',output={'error_type':type(exc).__name__})
        state.pop('skill_text',None)
        state.pop('workflow_materials',None)
        state['trace_path']=str(save_trace(state,self.config.trace_dir))
        state['checkpoint_path']=str(save_checkpoint(state,self.config.trace_dir))
        return state


def run_agent_workflow(packet,diagnostic_report=None,*,config=None,run_id=None,execute=False,**kwargs):
    return AgentRuntime(config,**kwargs).run(packet,diagnostic_report,run_id,execute)

__all__=['AgentRuntime','run_agent_workflow']
