"""Serializable research-influence control with explicit missing-data semantics."""
from copy import deepcopy
import math


def initial_state(config):
    return dict(weights={h['hypothesis_id']:h['evidence_weight']['initial'] for h in config['hypotheses']},
        caps={h['hypothesis_id']:h['evidence_weight']['cap'] for h in config['hypotheses']},
        strength=config['category_strength'],initial_strength=config['category_strength'],
        bad_streak=0,good_streak=0,no_effect_streak=0,reviews=0,applied_steps=0,
        rejected_steps=0,realized_current_changes=0,realized_endpoint_changes=0,cumulative_kl=0.,events=[])


def update(state,evidence,policy,*,dynamic=True):
    from molsteer.molthinker.research_rewards import revise_influence
    s=deepcopy(state);before=dict(weights=dict(s['weights']),strength=s['strength']);s['reviews']+=1
    dv=evidence.get('vina_delta');ds=evidence.get('strain_delta')
    known=all(isinstance(x,(float,int)) and math.isfinite(x) for x in (dv,ds))
    invalid=evidence.get('invalid',False)
    bad=invalid or (known and (dv>policy['bad_delta_vina'] or ds>policy['bad_delta_strain']))
    good=known and not bad and evidence.get('endpoint_graph_changed',False) and dv<=policy['good_delta_vina']
    no_effect=known and not bad and not evidence.get('endpoint_graph_changed',False) and abs(dv)<.01
    s['bad_streak']=s['bad_streak']+1 if bad else 0;s['good_streak']=s['good_streak']+1 if good else 0
    s['no_effect_streak']=s['no_effect_streak']+1 if no_effect else 0
    action='hold';reason='insufficient independent evidence' if not known and not invalid else 'within declared evidence tolerance'
    if dynamic:
        if invalid or s['bad_streak']>=policy['patience']:
            action='downweight_and_reduce_strength';reason='invalid proposal or repeated paired counterevidence';factor=.5
            s['weights']=revise_influence(s['weights'],s['caps'],factor);s['strength']*=factor;s['bad_streak']=0
        elif s['good_streak']>=policy['patience']:
            action='increase_within_evidence_cap';reason='repeated realized graph benefit under paired checks'
            s['weights']=revise_influence(s['weights'],s['caps'],1.2);s['strength']=min(s['initial_strength']*2,s['strength']*1.1);s['good_streak']=0
        elif s['no_effect_streak']>=policy['no_effect_patience']:
            action='downweight_no_realized_effect';reason='repeated local observations show no graph/score effect; do not equate probability ascent with potency'
            s['weights']=revise_influence(s['weights'],s['caps'],.8);s['no_effect_streak']=0
    else:reason='fixed-influence ablation; observations recorded without adaptive revision'
    event=dict(time=evidence.get('time'),action=action,reason=reason,evidence=deepcopy(evidence),before=before,
               after=dict(weights=dict(s['weights']),strength=s['strength']),route='MolThinker+MolExecutor' if action!='hold' else 'monitor_only',
               operator='Deterministic evidence policy; not an unobserved language-model call')
    s['events'].append(event)
    return s,event
