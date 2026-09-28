"""Optional binding of researcher evidence to an existing executable reward."""
from copy import deepcopy
from pathlib import Path
from molsteer.common import digest,file_hash
from .research import load_packet


def revise_influence(weights,caps,factor):
    """Resolve a monitor request against literature-derived influence ceilings."""
    if not 0<=factor<=2:raise ValueError('Invalid bounded research revision factor')
    return {key:min(caps[key],max(0.,value*factor)) for key,value in weights.items()}


def attach_research(program,packet_path,selected_ids,*,mode='shadow',dynamic=True,allow_exploratory=False):
    if mode not in ('off','shadow','active'):raise ValueError('Research mode must be off, shadow or active')
    if mode=='off':return deepcopy(program)
    if program.get('evaluator')!='outcome_aware':raise ValueError('Research execution currently requires the outcome-aware adapter contract')
    data=load_packet(packet_path);expected=program['native_reference']['sha256']
    if data['bindings'].get('native_reference_sha256')!=expected or data['bindings'].get('state_packet_id')!=program['packet_id']:
        raise ValueError('Research and reward evidence lineage mismatch')
    by={h['hypothesis_id']:h for h in data['hypotheses']};selected=[]
    if len(set(selected_ids))!=len(selected_ids):raise ValueError('Duplicate selected hypotheses')
    for key in selected_ids:
        if key not in by:raise ValueError('Unknown research hypothesis')
        h=by[key]
        if h['status']=='deferred' or not h.get('assignments'):raise ValueError('Selected hypothesis is not executable')
        if h['status']=='exploratory_transfer_only' and not allow_exploratory:raise ValueError('Exploratory transfer needs explicit experimental selection')
        selected.append(deepcopy(h))
    if mode=='active' and not selected:raise ValueError('Active research needs selected hypotheses')
    p=deepcopy(program);parent=p.pop('program_id');p['parent_program_id']=parent
    p['research']=dict(mode=mode,packet=dict(path=str(Path(packet_path).resolve()),sha256=file_hash(packet_path),packet_id=data['packet_id']),
        hypotheses=selected,dynamic=dynamic,category_strength=.2,max_kl=.03,max_log_change=3.,max_cumulative_kl=1.,
        review_every=5,monitor=dict(bad_delta_vina=.10,bad_delta_strain=2.,good_delta_vina=-.02,patience=2,no_effect_patience=3),
        control_scope='Mapped categorical probabilities before native sampling, with matched-RNG local checks and fresh endpoint assessment',
        activation='Explicit experiment selection; no literature potency assigned to this molecule')
    p['research_reward']='J(P,t) = sum_h lambda_h(t) * mean_{a in mapped h} log P_a(h_a); probability mirror ascent with categorical KL limits'
    p['research_derivation']=dict(observation='Residual native-clean buried polarity motivates a chemistry hypothesis; native self-repair is not credited.',
        evidence='Same-target sources establish research context. Exact transformation evidence may be absent and reduces the influence cap.',
        hypothesis='Mapped alternatives are experimental preferences; full sampled chemistry and external scores can disconfirm them.',
        choice='Keep the validated coordinate reward; add a separately budgeted soft categorical objective. No head-to-category derivative is assumed.',
        feedback='Matched one-step counterevidence changes influence and category strength; it does not establish experimental efficacy.')
    p['program_id']='rp_'+digest(p)[:24]
    return p
