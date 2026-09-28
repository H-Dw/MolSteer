"""Live-head reward and graph-conditioned structural feasibility."""
import bisect
import math
import torch
from .program import MolecularReward,angle,balance


class AffinityStructureReward(MolecularReward):
    def __init__(self,program,baseline,receptor,vocabulary,control):
        super().__init__(program,baseline,receptor,vocabulary)
        self.time=.5
        self.control=sorted((float(t),float(v)) for t,v in control.items())
        if len(self.control)<2 or any(not math.isfinite(t+v) for t,v in self.control):
            raise ValueError('Missing finite, same-time control trajectory')
        self.control_times=[t for t,_ in self.control]
        if program['geometry_scope']=='all':self.region=set(range(len(self.x0)))

    def set_time(self,t):
        self.time=float(t)

    def control_value(self):
        t=self.time
        if t<self.control[0][0]-1e-5 or t>self.control[-1][0]+1e-5:
            raise ValueError('Control trajectory does not cover current time')
        j=min(max(bisect.bisect_right(self.control_times,t),1),len(self.control)-1)
        a,av=self.control[j-1];b,bv=self.control[j]
        return av+(bv-av)*min(max((t-a)/(b-a),0.),1.)

    def affinity_utility(self,pred):
        head=self.spec['affinity_head']
        value=pred.get('affinity',{}).get(head)
        if value is None or value.numel()!=1 or not torch.isfinite(value).all():
            raise ValueError('Missing finite live affinity head: '+head)
        delta=(value.reshape(())-self.control_value())/self.spec['affinity_scale']
        cap=self.spec['affinity_saturation']
        return cap*torch.tanh(delta/cap),value.reshape(())

    def pocket_distance(self,x):
        receptor=x.new_tensor([a['coords'] for a in self.receptor])
        if len(receptor)==0:raise ValueError('Missing pocket coordinates')
        return torch.cdist(x,receptor).min(-1).values

    def evaluate(self,pred):
        utility,affinity=self.affinity_utility(pred)
        if self.spec['retain_initial']:
            structural,detail=super().evaluate(pred)
        else:
            x=pred['coords'];chem=self.chemistry(pred)
            if not chem['geometries']:raise ValueError('Missing graph-conditioned geometry references')
            if chem['missing_geometry_count']:raise ValueError('Incomplete graph-conditioned geometry references')
            protein,intra=self.overlaps(x,chem)
            residual=[];soft=[]
            for kind,ids,ref,tol in chem['geometries']:
                value=angle(x,ids) if kind=='angle' else (x[ids[0]]-x[ids[1]]).norm()
                normalized=(value-ref)/tol
                residual.append(torch.relu(normalized.abs()-1).square())
                soft.append(normalized.square())
            if self.spec['geometry_scope']=='all':
                geometry=torch.sqrt(1+torch.stack(residual).mean())-1
                geometry=geometry+self.spec['geometry_soft_weight']*torch.stack(soft).mean()
            else:
                geometry=torch.sqrt(1+torch.stack(residual).sum())-1
            clash=torch.maximum(protein.max(),intra.max()).clamp(min=0).square()/self.spec['clash_scale_angstrom']**2
            pocket=(torch.relu(self.pocket_distance(x)-self.spec['pocket_contact_distance'])/self.spec['pocket_distance_scale']).square().mean()
            structural=-self.spec['structure_weight']*balance([geometry,clash],self.spec['weights'],self.spec['tau'],self.spec['rho'])-self.spec['pocket_weight']*pocket
            detail=dict(components=dict(geometry=float(geometry.detach()),clash=float(clash.detach()),pocket=float(pocket.detach())),
                graph_cost=0.,graph_changed=chem['signature']!=self.reference_graph,smiles=chem['smiles'],
                max_protein_overlap=float(protein.max().detach()),max_intra_overlap=float(intra.max().detach()),
                graph_reference_count=len(chem['geometries']))
        reward=structural+self.spec['affinity_weight']*utility
        detail.update(affinity_head=self.spec['affinity_head'],affinity=float(affinity.detach()),
            control_affinity=self.control_value(),affinity_utility=float(utility.detach()),evaluation_time=self.time,
            structure_reward=float(structural.detach()))
        return reward,detail

    def feasible(self,candidate,reference):
        if self.spec['retain_initial']:
            return super().feasible(candidate,reference)
        failures=[]
        try:
            cc,bc=self.chemistry(candidate),self.chemistry(reference)
            cp,ci=self.overlaps(candidate['coords'],cc);bp,bi=self.overlaps(reference['coords'],bc)
            tol=self.spec['severe_overlap_angstrom']
            if ((cp>tol)&(cp>bp+1e-5)).any() or ((ci>tol)&(ci>bi+1e-5)).any():
                failures.append('new_or_worsened_severe_clash')
            if self.pocket_distance(candidate['coords']).max()>self.spec['max_atom_pocket_distance']:
                failures.append('outside_target_pocket')
        except ValueError as exc:failures.append(str(exc))
        return failures
