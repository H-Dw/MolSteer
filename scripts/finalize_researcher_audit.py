"""Audit saved-pose evaluation and unchanged batch members for the bounded study."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from molsteer.common import file_hash,write_json
from audit_independent_evaluation import pdb_atoms


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--prior',required=True)
    ap.add_argument('--tests-log',required=True,help='Test log to include in the audit')
    a=ap.parse_args()
    root=Path(a.root).resolve();prior=Path(a.prior).resolve();ev=root/'evaluation'
    data=json.loads((ev/'ReaderComparisonContext.json').read_text());checks=[]
    for row in data['observations']:
        source=Path(row['source_sdf']);d=ev/'cases'/row['label'];m=Chem.MolFromMolFile(str(source),removeHs=True)
        assert m is not None and file_hash(source)==row['source_sha256']
        shutil.copy2(source,d/'source_ligand.sdf')
        xyz=m.GetConformer().GetPositions()
        lines=(d/'vina/ligand.pdbqt').read_text().splitlines()
        q=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in lines
                    if l.startswith(('ATOM  ','HETATM')) and l.split()[-1] not in ('H','HD','HS') and not l.split()[-1].startswith('G')])
        assert len(xyz)==len(q)
        distances=np.linalg.norm(xyz[:,None]-q[None,:],axis=-1);i,j=linear_sum_assignment(distances);rounding=float(distances[i,j].max())
        assert rounding<=.000867
        ff=row['forcefield'];err=abs(sum(ff['component_strain'].values())-ff['strain_kcal_mol'])
        assert err<1e-6 and ff['minimization_status']==ff['hydrogen_relaxation_status']==0
        checks.append(dict(label=row['label'],source_unchanged=True,pdbqt_rounding_max_angstrom=rounding,
                           component_conservation_error=err,separate_optimization_copies=all(v['modes_separate'] for v in row['docking'].values())))
    x=pdb_atoms(ev/'preparation/receptor_aligned.pdb');y=pdb_atoms(ev/'preparation/receptor_prepared.pdb')
    assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
    control=torch.load(prior/'continuations/control/unguided/resume_final.pt',map_location='cpu',weights_only=True);batch=[]
    for label in ['no_projection','rebalanced_no_projection']:
        state=torch.load(root/'continuations'/label/'creativity/resume_final.pt',map_location='cpu',weights_only=True)
        unchanged=all(torch.equal(control['curr'][k][:2],state['curr'][k][:2]) for k in ['coords','atomics','bonds','charges'])
        assert unchanged
        batch.append(dict(label=label,other_batch_members_unchanged=unchanged))
    probe=json.loads((root/'DiscreteGradientAudit.json').read_text());assert probe['passed']
    write_json(root/'audit.json',dict(passed=True,cases=checks,receptor_heavy_atoms_unchanged=len(x),batch=batch,
        discrete_test_scope='Local probability derivative and one native sampling step; no claim of a medicinal full-rollout graph improvement'))
    shutil.copy2(Path(a.tests_log).resolve(),root/'tests.log')
    print('AUDIT PASSED')


if __name__=='__main__':main()
