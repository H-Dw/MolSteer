"""Defect-localized, staged reward for persistent molecular risks."""
import bisect
import json
import math
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from molsteer.common import file_hash
from .chemistry import decode_endpoint, signature
from .outcome_reward import OutcomeAwareReward
from .program import angle


def region_halo(mol, core):
    halo=set(int(i) for i in core)
    for i in tuple(halo):
        halo.update(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors())
    return tuple(sorted(halo))


def local_geometry(x, mol, core, bond_fraction, angle_degrees, tau):
    """Graph-conditioned flat-bottom residuals touching the diagnosed core."""
    core=set(int(i) for i in core);halo=set(region_halo(mol,core))
    h=Chem.AddHs(Chem.Mol(mol),addCoords=True)
    props=AllChem.MMFFGetMoleculeProperties(h,mmffVariant='MMFF94s')
    if props is None:raise ValueError('MMFF geometry parameters unavailable')
    rows=[];missing=0
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if not ({i,j}&core) or not {i,j}<=halo:continue
        p=props.GetMMFFBondStretchParams(h,i,j)
        if not p:missing+=1;continue
        ref=float(p[2]);tol=ref*bond_fraction;value=(x[i]-x[j]).norm()
        residual=torch.relu((value-ref).abs()-tol).div(tol).square()
        rows.append((residual,dict(kind='bond',atom_ids=[i,j],value=float(value.detach()),reference=ref,tolerance=tol)))
    for center in halo:
        neighbors=sorted(a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors() if a.GetIdx() in halo)
        for ii,i in enumerate(neighbors):
            for k in neighbors[ii+1:]:
                if not ({i,center,k}&core):continue
                p=props.GetMMFFAngleBendParams(h,i,center,k)
                if not p:missing+=1;continue
                ref=float(p[2]);value=angle(x,[i,center,k])
                residual=torch.relu((value-ref).abs()-angle_degrees).div(angle_degrees).square()
                rows.append((residual,dict(kind='angle',atom_ids=[i,center,k],value=float(value.detach()),reference=ref,tolerance=angle_degrees)))
    if missing:raise ValueError(f'Incomplete local MMFF geometry parameters: {missing}')
    if not rows:raise ValueError('No local MMFF geometry relations')
    residuals=torch.stack([v for v,_ in rows])
    loss=tau*(torch.logsumexp(residuals/tau,0)-math.log(len(rows)))
    detail=[]
    for residual,row in rows:
        row['residual']=float(residual.detach());detail.append(row)
    return loss,dict(relations=detail,relation_count=len(rows),max_residual=float(residuals.max().detach()),halo=sorted(halo))


class LocalMMFFRelaxation:
    """MMFF envelope derivative for a core plus one-bond halo.

    Heavy atoms outside the halo are fixed in the local reference relaxation.
    The derivative is supported only on halo heavy atoms.
    """
    def __init__(self):
        self.cache={}

    def evaluate(self,mol,coordinates,core):
        xyz=np.asarray(coordinates,dtype=np.float64);key=(signature(mol),tuple(sorted(core)),xyz.tobytes())
        if key in self.cache:return self.cache[key]
        m=Chem.Mol(mol);n=m.GetNumAtoms()
        if xyz.shape!=(n,3):raise ValueError('Coordinate/graph size mismatch')
        for i,row in enumerate(xyz):m.GetConformer().SetAtomPosition(i,row)
        halo=region_halo(m,core);h=Chem.AddHs(m,addCoords=True)
        if not AllChem.MMFFHasAllMoleculeParams(h):raise ValueError('MMFF parameters unavailable')
        props=AllChem.MMFFGetMoleculeProperties(h,mmffVariant='MMFF94s')
        fixed=AllChem.MMFFGetMoleculeForceField(h,props)
        for i in range(n):fixed.AddFixedPoint(i)
        status_h=fixed.Minimize(maxIts=1000,forceTol=1e-5,energyTol=1e-8)
        if status_h:raise ValueError('Hydrogen minimization did not converge')
        current=AllChem.MMFFGetMoleculeForceField(h,props)
        energy=float(current.CalcEnergy());gradient=np.asarray(current.CalcGrad()).reshape(-1,3)[:n]
        relaxed=Chem.Mol(h);relaxed_props=AllChem.MMFFGetMoleculeProperties(relaxed,mmffVariant='MMFF94s')
        local=AllChem.MMFFGetMoleculeForceField(relaxed,relaxed_props)
        halo_set=set(halo)
        for i in range(n):
            if i not in halo_set:local.AddFixedPoint(i)
        status_local=local.Minimize(maxIts=2500,forceTol=1e-5,energyTol=1e-8)
        if status_local:raise ValueError('Local MMFF relaxation did not converge')
        local_energy=float(local.CalcEnergy());strain=energy-local_energy
        if strain < -1e-4:raise ValueError('Local relaxation energy exceeded current energy')
        masked=np.zeros_like(gradient);masked[list(halo)]=gradient[list(halo)]
        result=dict(strain=max(0.,strain),energy=energy,local_reference=local_energy,gradient=masked,
            halo=list(halo),hydrogen_status=status_h,local_status=status_local)
        if len(self.cache)>=256:self.cache.clear()
        self.cache[key]=result
        return result

    def tensor(self,x,mol,core):
        result=self.evaluate(mol,x.detach().cpu().numpy(),core)
        value=x.new_tensor(result['strain']);gradient=x.new_tensor(result['gradient'])
        return _Envelope.apply(x,value,gradient),result


class _Envelope(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,value,gradient):
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    def backward(ctx,upstream):
        gradient,=ctx.saved_tensors
        return upstream*gradient,None,None


class LocalFirstReward(OutcomeAwareReward):
    """Lexicographic local geometry -> local strain -> affinity controller."""
    def __init__(self,program,baseline,receptor,vocabulary,control):
        super().__init__(program,baseline,receptor,vocabulary,control)
        cfg=program['local_first'];self.local_cfg=cfg;self.core=tuple(program['region_atom_ids'])
        path=Path(program['local_native_reference']['path'])
        if file_hash(path)!=program['local_native_reference']['sha256']:raise ValueError('Local native reference changed')
        ref=json.loads(path.read_text());self.local_frames=sorted(ref['frames'],key=lambda v:v['time'])
        self.local_times=[v['time'] for v in self.local_frames];self.local_mmff=LocalMMFFRelaxation()
        first=next(v for v in self.local_frames if v.get('status')=='ok')
        self.initial_signature=first['graph_signature'];self.last_signature=None;self.last_observation_time=None;self.persistence_count=0
        self.persistence_gate=False

    def native_local(self):
        j=min(max(bisect.bisect_right(self.local_times,self.time),1),len(self.local_frames)-1)
        a,b=self.local_frames[j-1],self.local_frames[j]
        row=a if abs(a['time']-self.time)<=abs(b['time']-self.time) else b
        if abs(row['time']-self.time)>.002:raise ValueError('Local native reference time is not matched')
        if row.get('status')!='ok':raise ValueError('Local native reference unavailable: '+row.get('reason','unknown'))
        return row

    def observe_graph(self,pred,observation_time=None):
        mol=decode_endpoint(pred,self.vocab);sig=signature(mol)
        same_time=(observation_time is not None and self.last_observation_time is not None and
            abs(float(observation_time)-float(self.last_observation_time))<1e-8)
        if sig!=self.last_signature:self.last_signature=sig;self.persistence_count=1
        elif not same_time:self.persistence_count+=1
        if observation_time is not None:self.last_observation_time=float(observation_time)
        self.persistence_gate=(sig!=self.initial_signature and self.persistence_count>=self.local_cfg['persistence_frames'])
        return dict(signature=sig,persistence_count=self.persistence_count,persistence_gate=self.persistence_gate,
            initial_self_replacing_branch=sig==self.initial_signature)

    def restore_controller(self,state):
        self.last_signature=state.get('last_signature');self.persistence_count=int(state.get('persistence_count',0))
        self.last_observation_time=state.get('last_observation_time')
        self.persistence_gate=bool(state.get('persistence_gate',False))

    def controller_state(self):
        return dict(last_signature=self.last_signature,last_observation_time=self.last_observation_time,
            persistence_count=self.persistence_count,persistence_gate=self.persistence_gate)

    def values(self,pred):
        mol=decode_endpoint(pred,self.vocab);x=pred['coords'];sig=signature(mol)
        geom,geom_detail=local_geometry(x,mol,self.core,self.local_cfg['bond_tolerance_fraction'],
            self.local_cfg['angle_tolerance_degrees'],self.local_cfg['tau_region'])
        strain_raw,strain_detail=self.local_mmff.tensor(x,mol,self.core)
        strain=torch.log1p(strain_raw/self.local_cfg['strain_scale_kcal_mol'])
        affinity,pk=self.affinity_utility(pred);native=self.native_local()
        threshold=max(0.,native['local_strain_kcal_mol']-self.local_cfg['minimum_extra_improvement_kcal_mol'])
        if self.persistence_gate and float(geom.detach())>self.local_cfg['geometry_zero_tolerance']:
            stage='geometry';active=-geom
        elif float(strain_raw.detach())>threshold:
            stage='strain';active=-strain
        else:
            stage='affinity';active=affinity
        detail=dict(stage=stage,graph_signature=sig,persistence_gate=self.persistence_gate,
            local_geometry=float(geom.detach()),local_strain=float(strain_raw.detach()),
            normalized_local_strain=float(strain.detach()),native_local_strain=native['local_strain_kcal_mol'],
            strain_unlock_threshold=threshold,affinity=float(pk.detach()),control_affinity=self.control_value(),
            affinity_utility=float(affinity.detach()),smiles=Chem.MolToSmiles(mol),geometry=geom_detail,
            local_relaxation={k:v for k,v in strain_detail.items() if k!='gradient'})
        return dict(geometry=geom,strain=strain,strain_raw=strain_raw,affinity=affinity,active=active),detail

    def gradient(self,pred,source_coordinates):
        values,detail=self.values(pred);stage=detail['stage'];logs={}
        if stage in ('geometry','strain'):
            gradient,=torch.autograd.grad(values['active'],source_coordinates,retain_graph=False,allow_unused=False)
            logs['projection']='not_applicable'
        else:
            gradient,=torch.autograd.grad(values['affinity'],source_coordinates,retain_graph=True,allow_unused=False)
            constraints=[('strain',values['strain'])]
            if self.persistence_gate:constraints.insert(0,('geometry',values['geometry']))
            grads=[]
            for name,value in constraints:
                g,=torch.autograd.grad(value,source_coordinates,retain_graph=True,allow_unused=False);grads.append((name,g))
            projection=[]
            for _ in range(4):
                changed=False
                for name,g in grads:
                    dot=(gradient*g).sum();denom=g.square().sum().clamp(min=1e-20)
                    if float(dot.detach())>0:
                        gradient=gradient-dot/denom*g;changed=True
                        projection.append(dict(constraint=name,removed_dot=float(dot.detach())))
                if not changed:break
            logs['projection']=projection
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite local-first gradient')
        logs.update(stage=stage,norm=float(gradient.detach().norm()),detail=detail)
        return gradient.detach(),logs

    def editable_mask(self,pred):
        mol=decode_endpoint(pred,self.vocab);mask=torch.zeros(len(pred['coords']),dtype=torch.bool,device=pred['coords'].device)
        mask[list(region_halo(mol,self.core))]=True
        return mask

    def evaluate(self,pred):
        values,detail=self.values(pred)
        return values['active'],detail

    def compare(self,candidate,base):
        cv,cd=self.values(candidate);bv,bd=self.values(base);order={'geometry':0,'strain':1,'affinity':2}
        if order[cd['stage']]<order[bd['stage']]:return False,-math.inf,'stage_regression',cd,bd
        if order[cd['stage']]>order[bd['stage']]:return True,1.,'stage_advance',cd,bd
        key={'geometry':'geometry','strain':'strain','affinity':'affinity'}[bd['stage']]
        if key=='affinity':gain=float(cv['affinity'].detach()-bv['affinity'].detach())
        else:gain=float(bv[key].detach()-cv[key].detach())
        return gain>self.local_cfg['minimum_same_time_gain'],gain,'same_stage_gain' if gain>0 else 'no_same_time_gain',cd,bd

    def feasible(self,candidate,reference):
        failures=super().feasible(candidate,reference)
        try:
            cv,cd=self.values(candidate);bv,bd=self.values(reference)
            cmol=decode_endpoint(candidate,self.vocab);bmol=decode_endpoint(reference,self.vocab)
            chalo=set(region_halo(cmol,self.core));outside=[i for i in range(len(candidate['coords'])) if i not in chalo]
            if outside:
                shift=(candidate['coords'][outside]-reference['coords'][outside]).norm(dim=-1).max()
                if float(shift)>self.local_cfg['max_endpoint_outside_halo_delta_angstrom']:
                    failures.append('outside_halo_endpoint_displacement')
            if signature(cmol)!=signature(bmol):
                if cd['local_geometry']>self.local_cfg['graph_change_max_geometry_loss']:
                    failures.append('graph_change_local_geometry')
                if cd['local_strain']>bd['local_strain']+self.local_cfg['graph_change_strain_allowance_kcal_mol']:
                    failures.append('graph_change_local_strain')
            elif cd['local_geometry']>bd['local_geometry']+self.local_cfg['geometry_regression_allowance']:
                failures.append('local_geometry_regression')
            threshold=cd['strain_unlock_threshold']+self.local_cfg['strain_regression_allowance_kcal_mol']
            if bd['stage']=='affinity' and cd['local_strain']>threshold:
                failures.append('repaired_local_strain_regression')
        except (ValueError,RuntimeError) as exc:failures.append(str(exc))
        return failures
