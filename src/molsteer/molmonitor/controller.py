"""Model-neutral bounded probe search and persistent event routing."""
from dataclasses import dataclass,asdict
import math


@dataclass(frozen=True)
class MonitorPolicy:
    initial_eta: float=30.
    min_eta: float=.01
    max_eta: float=10000.
    max_trials: int=9
    growth: float=4.
    persistence: int=3
    cooldown: int=5
    max_revisions: int=2
    temporal_mad_scale: float=4.
    native_rate_multiplier: float=3.
    rate_floor: float=2.
    geometry_worsening: float=.025
    geometry_max_worsening: float=.1
    severe_geometry_z: float=3.
    strain_review_every: int=5
    strain_review_start: float=.75
    strain_allowance_kcal_mol: float=1.
    changed_graph_strain_per_heavy_atom: float=1.
    minimum_gain: float=1e-7

    def __post_init__(self):
        vals=asdict(self)
        if any(not math.isfinite(v) or v<0 for v in vals.values()):raise ValueError('Nonfinite or negative monitor policy')
        if not 0<self.min_eta<=self.initial_eta<=self.max_eta or self.growth<=1 or self.max_trials<3:
            raise ValueError('Invalid guidance search bounds')
        if min(self.persistence,self.cooldown,self.strain_review_every)<1:raise ValueError('Invalid event window')


class AdaptiveController:
    def __init__(self,policy,state=None):
        self.policy=policy
        self.state=state or dict(eta=policy.initial_eta,streak=0,healthy_streak=0,revisions=0,last_event_step=-999,history=[])

    def strengths(self,saturation_eta):
        cap=min(self.policy.max_eta,max(self.policy.min_eta,saturation_eta))
        center=min(self.state['eta'],cap)
        values={self.policy.min_eta,center,cap}
        for power in [-2,-1,1,2,3,4]:
            values.add(max(self.policy.min_eta,min(cap,center*self.policy.growth**power)))
        return sorted(values,reverse=True)[:self.policy.max_trials-1]+[self.policy.min_eta] if len(values)>self.policy.max_trials else sorted(values,reverse=True)

    def choose(self,trials):
        allowed=[t for t in trials if not t['failures'] and t['effective_l2']>1e-12]
        # Select the largest TESTED effective control; nominal eta plateaus tie
        # toward the smallest coefficient, not a meaningless larger number.
        return max(allowed,key=lambda x:(round(x['effective_l2'],9),x['gain'],-x['eta'])) if allowed else None

    def route(self,step,time,trials,selected,*,unavailable=None,sentinel=None,budget_exhausted=False):
        reasons=[];action='adjust_strength';destination='MolExecutor'
        if selected:
            previous=self.state['eta'];self.state['eta']=selected['eta']
            reasons.append('increase' if selected['eta']>previous else 'decrease' if selected['eta']<previous else 'hold')
            self.state['healthy_streak']+=1
        else:
            self.state['healthy_streak']=0
            self.state['eta']=max(self.policy.min_eta,self.state['eta']/self.policy.growth)
        failure_types=sorted({f for t in trials for f in t['failures']})
        persistent=bool(trials) and selected is None and not budget_exhausted
        if unavailable and time>=.75:persistent=True;reasons.append('late_unavailable_reward')
        if sentinel and sentinel.get('reward_quality_conflict'):
            persistent=True;reasons.append('reward_quality_conflict')
        if sentinel and sentinel.get('changed_graph_quality_review'):
            persistent=True;reasons.append('changed_graph_quality_requires_review')
        if persistent:self.state['streak']+=1
        elif selected:self.state['streak']=max(0,self.state['streak']-1)
        if budget_exhausted:
            action='stop_guidance';reasons.append('injection_budget_exhausted')
        elif unavailable:
            action='native_only';reasons.append('reward_not_applicable')
        elif selected is None:
            action='native_only';reasons.append('no_admissible_tested_strength')
        if self.state['streak']>=self.policy.persistence and step-self.state['last_event_step']>=self.policy.cooldown:
            if self.state['revisions']<self.policy.max_revisions:
                destination='MolThinker';action='request_reward_revision'
                self.state['revisions']+=1;self.state['last_event_step']=step;self.state['streak']=0
            else:
                action='stop_guidance';reasons.append('revision_budget_exhausted')
        record=dict(step=step,time=time,destination=destination,action=action,reasons=reasons,
            failures=failure_types,selected_eta=selected['eta'] if selected else 0.,
            selected_effective_l2=selected['effective_l2'] if selected else 0.)
        self.state['history']=(self.state['history']+[record])[-8:]
        return record
