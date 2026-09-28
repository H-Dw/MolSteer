"""World-frame endpoint observations; graph-dependent identities are explicit."""
from types import SimpleNamespace
import math
import numpy as np
import torch
from rdkit import Chem
from molreader.io import decode,load_config
from molreader.metrics._shared import mmff
from molsteer.common import digest
from molsteer.molexecutor.program import angle


def snapshot(pred,reward,time,*,strain=False):
    x=pred['coords'].detach()
    out=dict(time=float(time),view='predicted_endpoint',finite=bool(torch.isfinite(x).all()),
        valid=False,chemical_valid=None,features={},geometry={},affinity={},limitations=[])
    if not out['finite']:
        out['invalid_reason']='nonfinite_coordinates';return out
    atomic=pred['atomics'].detach().argmax(-1).cpu().tolist()
    charges=pred['charges'].detach().argmax(-1).cpu().tolist()
    bonds=pred['bonds'].detach().argmax(-1).cpu().tolist()
    out.update(graph_id=digest([atomic,charges,bonds]),coords=x.cpu().tolist(),
        atoms=[dict(atom_id=i,element=reward.vocab['atomic_tokens'][a],
            charge=reward.vocab['charge_tokens'][charges[i]],confidence=float(pred['atomics'][i].max().detach())) for i,a in enumerate(atomic)])
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            value=float((x[i]-x[j]).norm())
            out['features'][f'distance:{i}:{j}']=dict(value=value,scale=1.,unit='angstrom',atom_ids=[i,j],family='distance')
    for k,v in pred.get('affinity',{}).items():
        value=float(v.detach())
        out['affinity'][k]=value if math.isfinite(value) else None
    try:
        chemistry=reward.chemistry(pred)
        out['chemical_valid']=True
        if chemistry.get('missing_geometry_count',0):raise ValueError('incomplete_geometry_references')
        if not chemistry['geometries']:raise ValueError('missing_geometry_references')
        out.update(valid=True,smiles=chemistry['smiles'])
        protein,intra=reward.overlaps(x,chemistry)
        out['protein_overlap']=float(protein.max());out['intra_overlap']=float(intra.max())
        zs=[]
        for kind,ids,ref,tol in chemistry['geometries']:
            value=float(angle(x,ids)) if kind=='angle' else float((x[ids[0]]-x[ids[1]]).norm())
            z=(value-ref)/tol;zs.append(z)
            # Reference values and local identity bind a metric across graph changes.
            identity=[[(atomic[i],charges[i]) for i in ids],[[bonds[i][j] for j in ids] for i in ids],ref,tol]
            key=kind+':'+':'.join(map(str,ids))+':'+digest(identity)[:12]
            out['geometry'][key]=dict(value=value,reference=ref,tolerance=tol,z=z,atom_ids=ids,
                family=kind,unit='degree' if kind=='angle' else 'angstrom')
            out['features'][key]=dict(value=z,scale=1.,unit='normalized_reference_deviation',atom_ids=ids,family=kind)
        out['geometry_rms_z']=float(np.sqrt(np.mean(np.square(zs))))
        out['geometry_max_abs_z']=max(abs(z) for z in zs)
        out['geometry_outliers']=sum(abs(z)>1 for z in zs)
        if strain:
            orders=np.array([[reward.vocab['bond_orders'][v] for v in row] for row in bonds])
            _,mol,error=decode(x.cpu().numpy(),[a['element'] for a in out['atoms']],
                               [a['charge'] for a in out['atoms']],orders)
            context=SimpleNamespace(mol=mol,sanitize_error=error,coords=x.cpu().numpy(),cache={},config=load_config())
            try:
                value=mmff(context)
                out['strain']=dict(value=value['strain_proxy_kcal_mol'],unit='kcal/mol',
                    converged=value['hydrogen_relaxation_status']==0 and value['minimization_status']==0,
                    method='same-graph MMFF94s local relaxation on a copy')
            except Exception as exc:out['limitations'].append('strain_unavailable: '+str(exc))
    except ValueError as exc:
        out['invalid_reason']=str(exc)
        if 'Chemical validity' in str(exc) or 'Disconnected' in str(exc):out['chemical_valid']=False
    return out


def rates(previous,current):
    dt=current['time']-previous['time']
    if dt<=0:raise ValueError('Temporal comparison requires positive progress interval')
    result={}
    for key,v in current['features'].items():
        if key not in previous['features']:continue
        old=previous['features'][key]
        if old['unit']!=v['unit'] or old['scale']!=v['scale']:continue
        result[key]=dict(rate=(v['value']-old['value'])/(dt*v['scale']),atom_ids=v['atom_ids'],family=v['family'])
    return result
