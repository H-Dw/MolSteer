"""Matched native reference, with sparse-reference limitations kept explicit."""
import math
import numpy as np
from .features import rates


class ReferenceTrajectory:
    def __init__(self,bundle):
        self.bundle=bundle
        self.frames=sorted(bundle['frames'],key=lambda x:x['time'])
        if len(self.frames)<2 or any(b['time']<=a['time'] for a,b in zip(self.frames,self.frames[1:])):
            raise ValueError('Reference needs distinct, ordered observations')
        self.dense=max(b['time']-a['time'] for a,b in zip(self.frames,self.frames[1:]))<=.021
        self.changes=[rates(a,b) for a,b in zip(self.frames,self.frames[1:])]

    def frame(self,time):
        nearest=min(self.frames,key=lambda x:abs(x['time']-time))
        if abs(nearest['time']-time)>1e-5:
            raise ValueError('No exact reference frame; do not interpolate changing chemical graphs')
        return nearest

    def temporal_evidence(self,previous,current,*,mad_scale=4.,native_multiplier=3.,rate_floor=2.):
        change=rates(previous,current);t=current['time']
        index=min(range(len(self.frames)-1),key=lambda i:abs(self.frames[i+1]['time']-t))
        evidence=[]
        for key,value in change.items():
            reference=self.changes[index].get(key)
            if reference is None:continue
            history=[abs(x[key]['rate']) for x in self.changes[max(0,index-3):index+1] if key in x]
            median=float(np.median(history));mad=float(np.median(np.abs(np.array(history)-median)))
            limit=max(rate_floor,native_multiplier*abs(reference['rate']),median+mad_scale*1.4826*mad)
            ratio=abs(value['rate'])/limit
            if ratio>1:
                evidence.append(dict(feature=key,atom_ids=value['atom_ids'],family=value['family'],
                    observed_rate=value['rate'],native_rate=reference['rate'],limit=limit,ratio=ratio,
                    baseline_median=median,baseline_mad=mad,unit='normalized_units_per_progress'))
        return dict(events=sorted(evidence,key=lambda x:x['ratio'],reverse=True),
            max_ratio=max([e['ratio'] for e in evidence],default=0.),
            comparable_features=len(set(change)&set(self.changes[index])),
            coverage='dense_native_reference' if self.dense else 'sparse_phase_average_only',
            threshold_origin='uncalibrated native-envelope heuristic; not a statistical confidence bound')

    def coarse_summary(self):
        chosen=[self.frame(t) for t in [.5,.75,1.]]
        rows=[]
        for a,b in zip(chosen,chosen[1:]):
            delta=rates(a,b);families={}
            for family in ['distance','bond','angle']:
                values=[abs(v['rate']) for v in delta.values() if v['family']==family]
                families[family]=dict(comparable=len(values),median=float(np.median(values)) if values else None,
                    maximum=max(values) if values else None)
            rows.append(dict(start=a['time'],end=b['time'],families=families,graph_changed=a.get('graph_id')!=b.get('graph_id')))
        return dict(intervals=rows,warning='Two coarse slopes cannot calibrate instantaneous spikes, uncertainty, or an acceleration distribution')
