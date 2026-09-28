"""Typed directional contact and smooth buried-unsatisfied-polar proxies.

These terms are geometric hypotheses, not calibrated binding/solvation energy.
Protein hydrogens are fixed by preparation. Ligand donor direction is a
heavy-neighbor bisector approximation; solvent water is not represented.
"""
import math
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,RDConfig
from rdkit.Chem import ChemicalFeatures


def features(mol):
    factory=ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))
    roles={}
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() in ('Donor','Acceptor'):
            for i in feature.GetAtomIds():roles.setdefault(i,[]).append(feature.GetFamily())
    return roles


def prepare_protein(path,reference_xyz,radius=10.):
    mol=Chem.MolFromPDBFile(str(path),removeHs=False)
    if mol is None:raise ValueError('Prepared receptor cannot be sanitized')
    xyz=mol.GetConformer().GetPositions();pt=Chem.GetPeriodicTable();roles=features(mol)
    close=np.linalg.norm(xyz[:,None,:]-np.asarray(reference_xyz)[None,:,:],axis=-1).min(1)<radius
    donors=[];acceptors=[]
    for i,role in roles.items():
        if not close[i]:continue
        if 'Donor' in role:
            for h in mol.GetAtomWithIdx(i).GetNeighbors():
                if h.GetAtomicNum()==1:donors.append(dict(atom_id=i,hydrogen_id=h.GetIdx(),coords=xyz[i].tolist(),hydrogen=xyz[h.GetIdx()].tolist()))
        if 'Acceptor' in role:
            neighbors=[xyz[a.GetIdx()] for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum()>1]
            direction=xyz[i]-np.mean(neighbors,axis=0) if neighbors else np.zeros(3)
            acceptors.append(dict(atom_id=i,coords=xyz[i].tolist(),direction=direction.tolist()))
    heavy=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1]
    return dict(donors=donors,acceptors=acceptors,heavy_coords=xyz[heavy].tolist(),
        heavy_radii=[pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in heavy],
        coverage='single prepared state; direct contacts only; no water-mediated satisfaction')


def unit(v):return v/v.norm(dim=-1,keepdim=True).clamp(min=1e-8)


def away_from_neighbors(x,mol,i):
    neighbors=[a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum()>1]
    if not neighbors:return x[i]*0
    return unit(-unit(x[neighbors]-x[i]).sum(0))


def measure(x,mol,protein,*,samples=48,softness=.2,probe=1.4):
    roles=features(mol);n=len(x);scores=[];ids=sorted(roles)
    for i in ids:
        terms=[]
        if 'Acceptor' in roles[i] and protein['donors']:
            d=x.new_tensor([v['coords'] for v in protein['donors']]);h=x.new_tensor([v['hydrogen'] for v in protein['donors']])
            radius=(x[i]-d).norm(dim=-1)
            orientation=(unit(h-d)*unit(x[i]-h)).sum(-1).clamp(min=0).pow(4)
            terms.append(torch.exp(-.5*((radius-2.9)/.5).square())*orientation)
        if 'Donor' in roles[i] and protein['acceptors']:
            a=x.new_tensor([v['coords'] for v in protein['acceptors']])
            ad=unit(x.new_tensor([v['direction'] for v in protein['acceptors']]))
            radius=(a-x[i]).norm(dim=-1)
            donor=(unit(a-x[i])*away_from_neighbors(x,mol,i)).sum(-1).clamp(min=0).square()
            acceptor=(unit(x[i]-a)*ad).sum(-1).clamp(min=0).square()
            terms.append(torch.exp(-.5*((radius-2.9)/.5).square())*donor*acceptor)
        terms=torch.cat(terms) if terms else x.new_zeros(1)+x.sum()*0
        scores.append(1-torch.exp(torch.log1p(-.95*terms.clamp(0,1)).sum()))
    if not ids:
        zero=x.sum()*0
        return dict(contact=zero,desolvation=zero,satisfaction=[],polar_ids=[],buried_area=[])
    satisfaction=torch.stack(scores)
    pt=Chem.GetPeriodicTable();r=x.new_tensor([pt.GetRvdw(a.GetAtomicNum())+probe for a in mol.GetAtoms()])
    k=torch.arange(samples,device=x.device,dtype=x.dtype);z=1-2*(k+.5)/samples;phi=k*(math.pi*(3-math.sqrt(5)))
    sphere=torch.stack([torch.sqrt(1-z*z)*torch.cos(phi),torch.sqrt(1-z*z)*torch.sin(phi),z],-1)
    points=x[ids,None,:]+r[ids,None,None]*sphere[None,:,:]
    ligand_d=(points[:,:,None,:]-x[None,None,:,:]).norm(dim=-1)-r[None,None,:]
    exclude=torch.zeros((len(ids),n),dtype=torch.bool,device=x.device)
    exclude[torch.arange(len(ids),device=x.device),torch.tensor(ids,device=x.device)]=True
    log_vis=torch.nn.functional.logsigmoid(ligand_d/softness).masked_fill(exclude[:,None,:],0.).sum(-1)
    protein_xyz=x.new_tensor(protein['heavy_coords']);protein_r=x.new_tensor(protein['heavy_radii'])+probe
    d=(points[:,:,None,:]-protein_xyz[None,None,:,:]).norm(dim=-1)-protein_r[None,None,:]
    protein_vis=torch.nn.functional.logsigmoid(d/softness).sum(-1).exp()
    area=4*math.pi*r[ids].square()
    buried=area*(log_vis.exp()*(1-protein_vis)).mean(-1)
    # One saturated satisfaction per ligand atom, not an unbounded pair count.
    return dict(contact=satisfaction.sum(),desolvation=(buried/20.*(1-satisfaction)).sum(),
        satisfaction=satisfaction,polar_ids=ids,buried_area=buried)
