"""Declarative reward compilation/testing; no dynamic Python execution."""
from copy import deepcopy
import math
from langchain_core.tools import tool
from molsteer.common import digest
from molsteer.molthinker.planner import validate_spec
from molsteer.molexecutor.offline import run_offline_trial


def validate_and_test_reward(packet, spec, report=None, *, iterations=3, strength=1.):
    validate_spec(spec,packet,report)
    if packet['steering']['graph_signatures'] != spec['graph_signatures']: raise ValueError('graph provenance mismatch')
    if spec['coordinate_hashes'] != {k:v['coordinate_hash'] for k,v in packet['steering']['coordinate_snapshots'].items()}: raise ValueError('coordinate provenance mismatch')
    if not math.isfinite(strength) or not 0<strength<=1: raise ValueError('invalid strength')
    if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=20: raise ValueError('invalid test iterations')
    trials=[]
    for group in spec['reward_groups']:
        view=group['view']
        # This mobility is expressly a numerical demonstration, never a live mask.
        movable=sorted({a for t in spec['terms'] if t['view']==view for a in t['atom_ids']})
        trial=run_offline_trial(packet,spec,view,movable,iterations=iterations,learning_rate=.25*strength)
        if not all(trial[k] for k in ('fixed_atoms_unchanged','input_snapshot_unchanged','monotone_penalty_descent')) or not trial['numerical_gradient']['passed']:
            raise ValueError('reward numerical validation failed')
        trials.append(trial)
    return dict(kind='RewardValidation',passed=True,reward_id=spec['reward_id'],trials=trials,
                status='tested' if trials else 'no_supported_terms',
                limitation='Frozen-graph coordinate-copy checks only; not generator inference or chemical acceptance')


def executor_tools(packet,spec,report,*,strength=1.):
    result={}; tested={}
    @tool
    def inspect_reward_program() -> dict:
        """Read the validated declarative reward specification and supported evaluator families."""
        return {'reward_spec':deepcopy(spec),'families':['flat_bottom_distance','flat_bottom_angle','minimum_distance'],
                'repair_scope':'Choose bounded test iterations; Thinker owns objective changes. No Python execution.'}
    @tool
    def test_reward_program(iterations:int=3) -> dict:
        """Compile/evaluate the fixed specification and test finite-difference gradients, descent and immutability on coordinate copies."""
        validation=validate_and_test_reward(packet,spec,report,iterations=iterations,strength=strength)
        tested['validation']=validation
        return validation
    @tool
    def submit_tested_program() -> dict:
        """Submit only after the current program passes actual numerical tests."""
        if 'validation' not in tested: raise ValueError('test_reward_program must succeed first')
        result['validation']=tested['validation']
        return {'status':'accepted','validation':tested['validation']}
    return [inspect_reward_program,test_reward_program,submit_tested_program],result


class Executor:
    """Standalone checked executor; formal inference requires a host approval gate."""
    def __init__(self,*,inference_adapter=None,approve_inference=False,max_repairs=2,execution_fn=None):
        self.inference_adapter=inference_adapter; self.approve_inference=approve_inference
        self.max_repairs=max_repairs; self.execution_fn=execution_fn
    def run(self,packet,spec,*,report=None,request=None):
        request=request or {}
        if self.execution_fn is not None:
            validate_spec(spec, packet, report)
            result = self.execution_fn(packet, spec, request)
            if not isinstance(result, dict): raise ValueError('execution_fn must return an object')
            validation = result
        else:
            validation=validate_and_test_reward(packet,spec,report)
        if request.get('formal_inference'):
            if not self.approve_inference or not callable(self.inference_adapter): raise RuntimeError('approved inference adapter required')
            return self.inference_adapter(packet=deepcopy(packet),reward_spec=deepcopy(spec),execution_result=validation,request=deepcopy(request))
        return validation

__all__=['Executor','executor_tools','validate_and_test_reward']


def executor_node(state, executor):
    result = executor.run(state['packet'], state['reward_spec'], report=state.get('diagnostic_report'), request=state.get('execution_request'))
    state['execution_result']=result; state['route']='monitor'; state['status']='monitoring'; state['step']=state.get('step',0)+1
    from .trace import append_trace
    return append_trace(state,node='executor',kind='observation',summary='Completed bounded declarative reward test',output={'passed':result.get('passed'),'reward_id':result.get('reward_id')})

__all__.append('executor_node')
