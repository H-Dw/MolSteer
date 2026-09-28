import numpy as np
from rdkit import Chem
from ..core import result,need_coords,need_protein,Unavailable

def compute(ctx):
    x=need_coords(ctx);protein=need_protein(ctx);pt=Chem.GetPeriodicTable()
    if any(a is None for a in ctx.atoms):raise Unavailable('SASA needs known elements')
    indices=[i for i,a in enumerate(ctx.atoms) if a!='H']
    x=x[indices];atoms=[ctx.atoms[i] for i in indices]
    p=[a for a in protein if a['record_type']=='ATOM']
    if not len(x) or not p:raise Unavailable('SASA requires ligand and protein heavy atoms')
    y=np.array([a['coords'] for a in p]);probe=ctx.config['thresholds']['sasa_probe_angstrom'];n=int(ctx.config['thresholds']['sasa_points'])
    lr=np.array([pt.GetRvdw(pt.GetAtomicNumber(a))+probe for a in atoms]);pr=np.array([pt.GetRvdw(a['atomic_number'])+probe for a in p])
    k=np.arange(n);z=1-2*(k+0.5)/n;phi=k*np.pi*(3-np.sqrt(5));unit=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
    free=[];bound=[];rows=[]
    for i,(c,r) in enumerate(zip(x,lr)):
        dots=c+r*unit
        own=((dots[:,None,:]-x[None,:,:])**2).sum(-1)<(lr[None,:]**2-1e-8);own[:,i]=False
        exposed=~own.any(1)
        receptor=((dots[:,None,:]-y[None,:,:])**2).sum(-1)<pr[None,:]**2
        area=4*np.pi*r*r
        a=float(exposed.mean()*area);b=float((exposed&~receptor.any(1)).mean()*area)
        free.append(a);bound.append(b)
        rows.append(dict(atom_id=int(ctx.atom_ids[indices[i]]),isolated_sasa_angstrom2=a,complex_sasa_angstrom2=b,buried_sasa_angstrom2=a-b))
    return result({'isolated_ligand_sasa_angstrom2':sum(free),'complex_ligand_sasa_angstrom2':sum(bound),
                   'ligand_buried_sasa_angstrom2':sum(free)-sum(bound),'buried_fraction':(sum(free)-sum(bound))/sum(free) if sum(free)>0 else None,'per_atom':rows},
                  units={'area':'angstrom^2'},method='Deterministic Fibonacci-sphere Shrake-Rupley approximation using heavy-atom vdW radii',
                  thresholds={'probe_angstrom':probe,'points_per_atom':n},
                  notes=['Ligand-side burial, not total interface BSA. Heavy-atom approximation and cropped receptor. Clash overlap can inflate burial.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('burial_sasa')
