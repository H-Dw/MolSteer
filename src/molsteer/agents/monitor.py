"""Per-metric rolling median/MAD monitoring with retune-before-replan policy."""
from collections import deque
from statistics import median
import math
from numbers import Real

class RobustMonitor:
    def __init__(self, config=None, **kwargs):
        defaults = dict(window=8, warmup=4, threshold=4.5, persistence=2, recovery=3,
                        cooldown=2, max_retunes=2, decay=.5, min_strength=.01, max_strength=1.)
        if config is not None: defaults.update({k:getattr(config,k) for k in defaults})
        unknown = set(kwargs)-set(defaults)
        if unknown: raise ValueError("unknown monitor configuration")
        defaults.update(kwargs)
        for k,v in defaults.items():
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v): raise ValueError("finite monitor configuration required")
            if k in {"window","warmup","persistence","recovery","cooldown","max_retunes"} and (not isinstance(v,int) or v<0): raise ValueError("integer monitor configuration required")
        if not 1<=defaults['warmup']<=defaults['window'] or defaults['persistence']<1 or defaults['recovery']<1 or defaults['threshold']<=0 or not 0<=defaults['decay']<=1 or not 0<=defaults['min_strength']<=defaults['max_strength']<=1:
            raise ValueError("invalid monitor configuration")
        self.__dict__.update(defaults)
        self.reset()

    def reset(self):
        self.history = {}
        self.anomaly_streak = self.healthy_streak = self.retunes = self.cooldown_left = 0
        self.strength = self.max_strength
        self.last_event = {}

    def observe(self, metrics, *, step=None):
        if not isinstance(metrics,dict) or not metrics: raise ValueError("metrics must be nonempty")
        if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in metrics.values()):
            event = dict(kind="MonitorEvent", route="stop", action="hard_safety_stop", reason="nonfinite_or_invalid_metric", step=step)
            self.last_event=event
            return event
        scores={}; warm=[]
        for key,value in metrics.items():
            history=self.history.setdefault(key,deque(maxlen=self.window))
            if len(history)<self.warmup:
                warm.append(key)
                scores[key]=dict(score=0.,center=None,mad=None)
            else:
                center=median(history); mad=median(abs(x-center) for x in history)
                scale=max(1.4826*mad,1e-9*max(1.,abs(center)))
                scores[key]=dict(score=min(abs(value-center)/scale,1e100),center=center,mad=mad)
        anomaly=any(v['score']>=self.threshold for v in scores.values())
        # Never train a baseline on anomalies, including during retuning/cooldown.
        if not anomaly:
            for key,value in metrics.items(): self.history[key].append(float(value))
        blocked=self.cooldown_left>0
        if blocked: self.cooldown_left-=1
        route,action='stable','warmup' if warm else 'hold'
        if anomaly:
            self.anomaly_streak+=1; self.healthy_streak=0
            if self.anomaly_streak>=self.persistence and not blocked:
                self.anomaly_streak=0; self.cooldown_left=self.cooldown
                if self.retunes<self.max_retunes:
                    self.retunes+=1; self.strength=max(self.min_strength,self.strength*self.decay)
                    route,action='executor','retune_strength'
                else: route,action='thinker','revise_reward'
        else:
            self.anomaly_streak=0
            if not warm:
                self.healthy_streak+=1
                if self.healthy_streak>=self.recovery and not blocked:
                    self.strength=min(self.max_strength,self.strength+(self.max_strength-self.strength)*.25)
                    self.healthy_streak=0; action='recover'
        event=dict(kind='MonitorEvent',route=route,action=action,anomaly=anomaly,metrics=dict(metrics),scores=scores,
                   score=max(x['score'] for x in scores.values()),strength=self.strength,retunes=self.retunes,
                   cooldown_left=self.cooldown_left,history_sizes={k:len(v) for k,v in self.history.items()},step=step)
        self.last_event=event
        return event


def monitor_metrics(result):
    if isinstance(result.get('metrics'),dict): return result['metrics']
    keys=('penalty_after','delta_norm_angstrom','max_atom_displacement_angstrom','new_protein_clashes','new_bond_window_violations')
    metrics={k:result[k] for k in keys if k in result}
    if not metrics: raise ValueError('execution result has no monitoring metrics')
    return metrics

__all__=['RobustMonitor','monitor_metrics']
