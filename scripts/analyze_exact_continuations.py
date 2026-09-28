"""Compare fixed-reward early starts, graph edits, descriptors and affinity."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator,rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from molreader.io import load_stage
from molsteer.common import write_json


def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def graph_changes(ref,mol):
    atoms=[dict(atom_id=i,before=dict(element=a.GetSymbol(),charge=a.GetFormalCharge()),
                after=dict(element=b.GetSymbol(),charge=b.GetFormalCharge()))
           for i,(a,b) in enumerate(zip(ref.GetAtoms(),mol.GetAtoms()))
           if (a.GetSymbol(),a.GetFormalCharge())!=(b.GetSymbol(),b.GetFormalCharge())]
    edges=lambda m:{tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):b.GetBondTypeAsDouble() for b in m.GetBonds()}
    x,y=edges(ref),edges(mol)
    bonds=[dict(atom_ids=list(k),before=x.get(k,0),after=y.get(k,0)) for k in sorted(x.keys()|y.keys()) if x.get(k,0)!=y.get(k,0)]
    return dict(atoms=atoms,bonds=bonds,element_changes=sum(a['before']['element']!=a['after']['element'] for a in atoms),
                charge_changes=sum(a['before']['charge']!=a['after']['charge'] for a in atoms),bond_changes=len(bonds))


def chemical_role_mapping(ref,mol):
    """Match elements and bond orders while allowing protonation/charge changes.

    This separates slot rewiring from a genuinely different heavy-atom scaffold.
    Query atoms match atomic number only; all original heavy-atom bonds remain.
    """
    if ref.GetNumAtoms()!=mol.GetNumAtoms() or ref.GetNumBonds()!=mol.GetNumBonds():return dict(same_heavy_atom_graph=False)
    query=Chem.RWMol()
    for atom in ref.GetAtoms():query.AddAtom(Chem.AtomFromSmarts(f'[#{atom.GetAtomicNum()}]'))
    for bond in ref.GetBonds():query.AddBond(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondType())
    matches=mol.GetSubstructMatches(query.GetMol(),uniquify=False,maxMatches=10000)
    if not matches:return dict(same_heavy_atom_graph=False)
    mapping=min(matches,key=lambda m:sum(i!=j for i,j in enumerate(m)))
    changes=[]
    for i,j in enumerate(mapping):
        x,y=ref.GetAtomWithIdx(i),mol.GetAtomWithIdx(j)
        if (x.GetFormalCharge(),x.GetTotalNumHs())!=(y.GetFormalCharge(),y.GetTotalNumHs()):
            changes.append(dict(reference_atom=i,candidate_atom=j,element=x.GetSymbol(),
                before_charge=x.GetFormalCharge(),after_charge=y.GetFormalCharge(),before_H=x.GetTotalNumHs(),after_H=y.GetTotalNumHs()))
    return dict(same_heavy_atom_graph=True,reference_to_candidate=list(mapping),charge_or_hydrogen_changes=changes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve()
    runs=json.loads((root/'runs.json').read_text());quality=json.loads((root/'final_quality.json').read_text())
    if len(runs)!=12 or len(quality)!=12:raise ValueError('Expected ten fixed-RNG comparisons and two one-time SC ablations')
    configs={stage:json.loads((root/'configurations'/f'{stage}.json').read_text()) for stage in ['t_0.25','t_0.50']}
    program_hashes={s:hashfile(c['reward_programs']['creativity']) for s,c in configs.items()}
    reference_hashes={s:hashfile(Path(c['reward_reference_stage'])/'world_prediction.pt') for s,c in configs.items()}
    assert len(set(program_hashes.values()))==len(set(reference_hashes.values()))==1
    exact={s:json.loads((root/'validation'/f'{s}_unguided_exact.json').read_text()) for s in configs}
    assert all(v['all_exact'] for v in exact.values())
    baseline=next(r for r in quality if r['stage']=='t_0.50' and r['label']=='eta_0')
    ctx=load_stage(baseline['source']);ref=ctx.mol
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);ref_fp=fp.GetFingerprint(ref)
    rows=[];mols=[];legends=[]
    for row in sorted(quality,key=lambda r:(r['stage'],r['clear_initial_self_condition'],r['strength'])):
        r=dict(row);c=load_stage(r['source']);mol=c.mol
        r['same_graph_as_unguided']=r['smiles']==baseline['smiles']
        r['graph_edits']=graph_changes(ref,mol) if mol else None
        r['chemical_role_mapping']=chemical_role_mapping(ref,mol) if mol else None
        packet=json.loads((Path(r['directory'])/'evaluation'/r['arm']/'final/StatePacket.json').read_text())
        r['local_geometry_evidence']=[e for m in packet['observations'] if m['view']=='prediction' and m['metric_id']=='mmff_local_geometry' for e in m['evidence']]
        r['morgan_tanimoto_to_unguided']=DataStructs.TanimotoSimilarity(ref_fp,fp.GetFingerprint(mol)) if mol else None
        r['affinity_delta']={k:r['affinity'][k]-baseline['affinity'][k] for k in ['pic50','pki','pkd','pec50']}
        r['descriptor_delta']={k:r['descriptors'][k]-baseline['descriptors'][k] for k in r['descriptors'] if isinstance(r['descriptors'][k],(int,float))}
        r['same_graph_strain_reduction_percent']=(baseline['strain']-r['strain'])/baseline['strain']*100 if r['same_graph_as_unguided'] and r['strain_status']=='ok' else None
        displacement=np.linalg.norm(c.coords-ctx.coords,axis=-1)
        r['slot_rmsd_angstrom']=float(np.sqrt(np.mean(displacement**2)))
        r['slot_max_displacement_angstrom']=float(displacement.max())
        trace=[json.loads(line) for line in (Path(r['directory'])/r['arm']/'guidance_trace.jsonl').read_text().splitlines()]
        accepted=[x for x in trace if x.get('accepted') and x.get('injected_max_angstrom',0)>1e-8]
        failures=Counter(f for x in trace for attempt in x.get('proposal_attempts',[]) for f in attempt['failures'])
        early=[x for x in trace if x['step']<50]
        r['guidance']=dict(first_nonzero_accepted_step=min((x['step'] for x in accepted),default=None),
            accepted_nonzero_steps=len(accepted),accepted_nonzero_before_50=sum(x['step']<50 for x in accepted),
            max_path_used_before_50=early[-1]['injected_path_max_angstrom'] if early else 0,
            unavailable_steps=[dict(step=x['step'],reason=x['guidance_unavailable']) for x in trace if 'guidance_unavailable' in x],
            rejected_proposal_reasons=dict(failures))
        rows.append(r)
        if not r['clear_initial_self_condition'] and r['strength']>0:
            draw=Chem.Mol(mol);Chem.RemoveStereochemistry(draw);rdDepictor.Compute2DCoords(draw)
            mols.append(draw);legends.append(f"Start {r['stage'][2:]}, eta={r['strength']:g}\n"+r['descriptors']['formula'])
    result=dict(shared_reward_sha256=program_hashes,shared_X0_p0_sha256=reference_hashes,
        initial_time_comparison='Only execution start changes; reward, reference X0, p0, masks, eta grid and cumulative path cap fixed',
        interpretation='Retrospective application of a t=0.50 diagnostic reward to an earlier checkpoint; not a prospective online policy',
        exact_resume=exact,rows=rows)
    write_json(root/'comparison.json',result)
    keys=['stage','label','strength','same_graph_as_unguided','strain','same_graph_strain_reduction_percent','slot_rmsd_angstrom','morgan_tanimoto_to_unguided']
    extra=['pic50','pki','pkd','pec50','formula','mw','logp','tpsa','qed','hbd','hba','rotatable_bonds']
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys+extra);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in keys},**r['affinity'],**{k:r['descriptors'].get(k) for k in extra if k not in r['affinity']}})
    for kind in ['png','svg']:
        drawer=(rdMolDraw2D.MolDraw2DCairo if kind=='png' else rdMolDraw2D.MolDraw2DSVG)(1600,900,400,450)
        drawer.drawOptions().legendFontSize=20
        drawer.DrawMolecules(mols,legends=legends,highlightAtoms=[[1,10,14]]*len(mols))
        drawer.FinishDrawing();data=drawer.GetDrawingText();(root/f'guided_structures.{kind}').write_bytes(data.encode() if isinstance(data,str) else data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for stage,color in [('t_0.25','#007D8A'),('t_0.50','#B75435')]:
        selected=[r for r in rows if r['stage']==stage and not r['clear_initial_self_condition']]
        xs=range(len(selected));labels=[str(int(r['strength'])) for r in selected]
        for ax,key,label in [(axes[0],'strain','MMFF relaxation proxy (kcal/mol)'),(axes[1],'pki','Predicted pKi'),(axes[2],'pkd','Predicted pKd')]:
            ys=[r[key] if key=='strain' else r['affinity'][key] for r in selected]
            ax.plot(xs,[y if r['same_graph_as_unguided'] else float('nan') for r,y in zip(selected,ys)],'o-',color=color,label=f'Start {stage[2:]}')
            changed=[(i,y) for i,(r,y) in enumerate(zip(selected,ys)) if not r['same_graph_as_unguided']]
            if changed:ax.scatter([p[0] for p in changed],[p[1] for p in changed],marker='D',color=color)
            ax.set_xticks(list(xs),labels);ax.set_xlabel('Guidance strength eta');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].set_title('Diamonds: changed chemical identity',fontsize=10)
    axes[0].legend();fig.savefig(root/'start_time_comparison.png',dpi=180);fig.savefig(root/'start_time_comparison.svg');plt.close(fig)
    print(json.dumps([{k:r[k] for k in ['stage','label','strain','same_graph_as_unguided','affinity','guidance']} for r in rows],indent=2))


if __name__=='__main__':main()
