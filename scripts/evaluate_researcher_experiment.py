"""Unified saved-pose assessment, provenance checks and report materialization."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from molsteer.common import file_hash,write_json
from audit_independent_evaluation import pdb_atoms


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--prior',required=True);ap.add_argument('--preparation',required=True);a=ap.parse_args()
    root=Path(a.root).resolve();prior=Path(a.prior).resolve();cfg=json.loads((root/'base_execution.json').read_text());ev=root/'evaluation'
    shutil.copytree(a.preparation,ev/'preparation',dirs_exist_ok=True)
    sub=Path(cfg['target_id'])/f'ligand_{cfg["ligand_index"]:03d}'/'final'
    native=Path(cfg['saved_stage']).parent/'final'
    rows=[dict(label='native_final',path=str(native)),dict(label='legacy_fixed300',path=str(prior/'continuations/legacy_fixed300/creativity'/sub))]
    runs=json.loads((root/'runs.json').read_text())
    for run in runs:rows.append(dict(label=run['label'],path=str(root/'continuations'/run['label']/'creativity'/sub)))
    write_json(ev/'cases.json',rows)
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_guidance_independently.py')),
        '--model-root',cfg['model_root'],'--output',str(ev),'--cases-json',str(ev/'cases.json')],check=True)
    data=json.loads((ev/'ReaderComparisonContext.json').read_text());audit=[]
    for row in data['observations']:
        source=Path(row['source_sdf']);d=ev/'cases'/row['label'];mol=Chem.MolFromMolFile(str(source),removeHs=True)
        assert mol is not None and file_hash(source)==row['source_sha256'];shutil.copy2(source,d/'source_ligand.sdf')
        xyz=mol.GetConformer().GetPositions();lines=(d/'vina/ligand.pdbqt').read_text().splitlines()
        q=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in lines
                    if l.startswith(('ATOM  ','HETATM')) and l.split()[-1] not in ('H','HD','HS') and not l.split()[-1].startswith('G')])
        assert len(xyz)==len(q);dist=np.linalg.norm(xyz[:,None]-q[None,:],axis=-1);i,j=linear_sum_assignment(dist);rounding=float(dist[i,j].max());assert rounding<=.000867
        ff=row['forcefield'];error=abs(sum(ff['component_strain'].values())-ff['strain_kcal_mol']);assert error<1e-6
        assert ff['minimization_status']==ff['hydrogen_relaxation_status']==0
        audit.append(dict(label=row['label'],source_sha256=row['source_sha256'],pdbqt_rounding_max_angstrom=rounding,
            energy_conservation_error=error,separate_local_optimization_copy=True))
    x=pdb_atoms(ev/'preparation/receptor_aligned.pdb');y=pdb_atoms(ev/'preparation/receptor_prepared.pdb')
    assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
    write_json(root/'evaluation_audit.json',dict(passed=True,cases=audit,receptor_heavy_atoms_unchanged=len(x),source_structures_unchanged=True,
        scoring_roles='Vina is an external gate and final score, not fully held out. Vinardo is correlated; no experimental efficacy claim.'))
    summary=[]
    for row in data['observations']:
        label=row['label'];comparison=data['comparisons'][label]
        summary.append(dict(label=label,pkd=row.get('model_predictions',{}).get('affinity',{}).get('pkd'),
            vina=row['docking']['vina']['score_only_kcal_mol'],vinardo=row['docking']['vinardo']['score_only_kcal_mol'],
            strain=row['forcefield']['strain_kcal_mol'],same_graph=comparison['same_slot_graph'],same_stereo=comparison['same_isomeric_smiles'],
            posebusters=row['metrics']['posebusters']['values']['all_reported_checks_pass'],smiles=row['smiles']))
    write_json(root/'comparison.json',summary)
    import csv
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
