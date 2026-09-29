"""Receive monitor evidence without mistaking numerical alarms for diagnoses."""
from copy import deepcopy
from .knowledge import KnowledgeBase
from molsteer.common import digest


def prepare_revision_context(request,program,knowledge_path):
    if request['parent_program_id']!=program['program_id'] or request['packet_id']!=program['packet_id']:
        raise ValueError('Monitor handoff does not match the current program and evidence')
    failures=' '.join(request['event']['failures']+request['event']['reasons'])
    tags=['local_geometry','multiobjective']
    if 'strain' in failures or 'quality_conflict' in failures or 'changed_graph_quality' in failures:tags+=['conformational_strain']
    if 'clash' in failures:tags+=['protein_clashes']
    if 'Chemical' in failures or 'Disconnected' in failures:tags+=['chemical_validity','graph_connectivity']
    kb=KnowledgeBase(knowledge_path)
    hits=[h for h in kb.retrieve(tags) if h['retrieval_score']>0][:6]
    program_summary={k:v for k,v in program.items() if k not in ['terms','retrieval','source_packet','source_report','expert_spec']}
    if program.get('evaluator')=='agent_expert':
        program_summary['expert_handoff']={k:deepcopy(program['expert_spec'][k]) for k in
            ('biology_plan','mathematical_design','model_dynamics')}
        program_summary['next_action']='Acquire a fresh StatePacket and DiagnosticReport, then rerun biology and mathematics; do not reuse stale graph or gradient bindings'
    program_summary['term_count']=len(program.get('terms',[]))
    program_summary['full_program_digest']=digest(program)
    retrieval=[{k:h[k] for k in ['function_id','name_en','formula','role','prerequisites','source','retrieval_score','matched_categories']} for h in hits]
    return dict(kind='MolThinkerRevisionContext',request_id=request['request_id'],
        parent_program_id=program['program_id'],skill='molthinker-reward-creativity',
        evidence=request,program_summary=program_summary,knowledge_source=dict(path=str(knowledge_path),sha256=kb.sha256),
        retrieval=retrieval,
        task='Assess the monitor evidence, distinguish step-size failure from reward misspecification, and return a declarative reward revision with a testable rationale. Preserve evidence bindings and noncompensable constraints. Do not claim that a proposal was tested unless validation results are supplied.',
        status='ready_for_reasoning_agent; no language-model call has been fabricated')


def apply_revision_response(checkpoint,response):
    state=checkpoint['guidance_state'];pending=state.get('pending_request')
    if not pending or response['request_id']!=pending['request_id']:
        raise ValueError('Revision response does not match the pending request')
    if response['parent_program_id']!=state['program_id']:
        raise ValueError('Revision response has a stale parent program')
    program=response['program']
    resolution=response.get('resolution','revise_reward')
    if resolution not in ['revise_reward','stop_guidance']:raise ValueError('Unsupported revision resolution')
    if program['packet_id']!=pending['packet_id'] or program['identity']!=pending['identity']:
        raise ValueError('Revision must preserve molecular/evidence identity')
    if not response.get('rationale') or not response.get('validation_plan'):
        raise ValueError('Revision requires rationale and validation plan')
    for key,value in pending.get('guard_contract',{}).items():
        if program.get(key)!=value:raise ValueError('Revision silently changed a guard/reference contract: '+key)
    if resolution=='revise_reward' and program['program_id']==state['program_id']:
        raise ValueError('Revision must have a distinct program identity')
    if resolution=='stop_guidance' and program['program_id']!=state['program_id']:
        raise ValueError('Stopping guidance must retain the current program identity')
    # The downstream evaluator still validates supported terms/scales, and its
    # live finite-difference preflight is required before continuation.
    revised=deepcopy(checkpoint)
    revised['guidance_state']['program_id']=program['program_id']
    revised['guidance_state']['pending_request']=None
    revised['guidance_state']['monitor']['sentinel']=None
    revised['guidance_state']['monitor']['controller']['streak']=0
    if resolution=='stop_guidance':revised['guidance_state']['monitor']['stopped']=True
    revised['guidance_state'].setdefault('program_lineage',[]).append(dict(parent=response['parent_program_id'],
        program_id=program['program_id'],request_id=response['request_id'],resolution=resolution,response_digest=digest(response)))
    return revised,program
