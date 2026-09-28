"""Continuous, counterfactual molecular reward with explicit independent roles."""
import json
import math
from pathlib import Path
import torch
from rdkit import Chem
from .affinity import AffinityStructureReward
from .chemistry import decode_endpoint
from .mmff_bridge import MMFFStrain
from .interface_energy import measure
from molsteer.common import file_hash


class OutcomeObservables:
    def __init__(self,protein,references=None):
        self.protein=protein;self.mmff=MMFFStrain()
        self.mmff.references=dict(references or {})

    def evaluate(self,pred,vocabulary):
        mol=decode_endpoint(pred,vocabulary);x=pred['coords']
        strain=self.mmff.tensor(x,mol)
        interface=measure(x,mol,self.protein)
        return dict(strain=strain,contact=interface['contact'],desolvation=interface['desolvation']),mol,interface


class OutcomeAwareReward(AffinityStructureReward):
    def __init__(self,program,baseline,receptor,vocabulary,control):
        super().__init__(program,baseline,receptor,vocabulary,control)
        path=Path(program['native_reference']['path'])
        if file_hash(path)!=program['native_reference']['sha256']:raise ValueError('Native outcome reference changed')
        self.reference=json.loads(path.read_text());self.observables=OutcomeObservables(self.reference['protein'],self.reference['mmff_references'])

    def native(self):
        frame=min(self.reference['frames'],key=lambda row:abs(row['time']-self.time))
        if abs(frame['time']-self.time)>.002:raise ValueError('Native reference time is not matched')
        if frame['status']!='ok':raise ValueError('Native comparison observables unavailable at this time')
        return frame

    def components(self,pred):
        base=self.native();obs,mol,interface=self.observables.evaluate(pred,self.vocab)
        x=pred['coords'];s=self.spec['outcome_scales'];w=self.spec['outcome_weights']
        pk=pred.get('affinity',{}).get(self.spec['affinity_head'])
        if pk is None or pk.numel()!=1 or not torch.isfinite(pk):raise ValueError('Live primary affinity unavailable')
        affinity=2*torch.tanh((pk.reshape(())-base['affinity'][self.spec['affinity_head']])/2)
        strain=math.log1p(max(0.,base['observables']['strain'])/s['strain_kcal_mol'])-torch.log1p(obs['strain'].clamp(min=0)/s['strain_kcal_mol'])
        contact=(obs['contact']-base['observables']['contact'])/s['contact']
        desolvation=(base['observables']['desolvation']-obs['desolvation'])/s['desolvation']
        pocket=(torch.relu(self.pocket_distance(x)-self.spec['pocket_contact_distance'])/self.spec['pocket_distance_scale']).square().mean()
        parts=dict(affinity=w['affinity']*affinity,strain=w['strain']*strain,contact=w['contact']*contact,
            desolvation=w['desolvation']*desolvation,pocket=-w['pocket']*(pocket-base['pocket']))
        return parts,dict(observables={k:float(v.detach()) for k,v in obs.items()},
            native_observables=base['observables'],affinity=float(pk.detach()),
            control_affinity=base['affinity'][self.spec['affinity_head']],time=self.time,
            polar_ids=interface['polar_ids'],smiles=Chem.MolToSmiles(mol),
            continuous_strain_reference='frozen same-graph local minimum',
            proxy_limit='Directional contact and unsatisfied burial are geometry proxies, not binding free energy')

    def evaluate(self,pred):
        parts,detail=self.components(pred)
        value=sum(parts.values(),pred['coords'].sum()*0)
        detail['components']={k:float(v.detach()) for k,v in parts.items()}
        return value,detail


def conflict_aware_gradient(parts,coordinates,*,project=True):
    gradients={k:torch.autograd.grad(v,coordinates,retain_graph=True,allow_unused=False)[0] for k,v in parts.items()}
    structure=gradients['strain'];denom=structure.square().sum().clamp(min=1e-20)
    affinity=gradients['affinity'];cosine=float((affinity*structure).sum().detach()/(affinity.norm()*structure.norm()).detach().clamp(min=1e-20))
    projected={};log={}
    for key,g in gradients.items():
        dot=(g*structure).sum();conflicting=key!='strain' and float(dot.detach())<0
        projected[key]=g-dot/denom*structure if project and conflicting else g
        log[key]=dict(norm=float(g.norm().detach()),projected_norm=float(projected[key].norm().detach()),
            conflicted_with_strain=conflicting)
    combined=sum(projected.values()).detach()
    return combined,dict(affinity_structure_cosine=cosine,components=log,
        direction_is_raw_reward_gradient=not project,
        interpretation='First-order structure-protected search direction; not claimed to be the unchanged scalar reward gradient')
