from collections import defaultdict
import numpy as np
from ..core import result,need_mol,need_protein,plane

def compute(ctx):
    mol=need_mol(ctx);protein=need_protein(ctx);residues=defaultdict(dict)
    for a in protein:residues[(a['residue_id'],a['residue_name'])][a['atom_name']]=a
    templates={'PHE':[['CG','CD1','CE1','CZ','CE2','CD2']], 'TYR':[['CG','CD1','CE1','CZ','CE2','CD2']],
               'HIS':[['CG','ND1','CE1','NE2','CD2']], 'TRP':[['CG','CD1','NE1','CE2','CD2'],['CD2','CE2','CZ2','CH2','CZ3','CE3']]}
    rings=[r for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    cutoff=ctx.config['thresholds']['ring_center_distance_angstrom'];rows=[]
    for ring in rings:
        c,n,_=plane(ctx.coords[list(ring)])
        for (resid,name),atoms in residues.items():
            for names in templates.get(name,[]):
                if not all(a in atoms for a in names):continue
                pc,pn,_=plane(np.array([atoms[a]['coords'] for a in names]));delta=pc-c;distance=float(np.linalg.norm(delta))
                if distance>cutoff:continue
                angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(n,pn)),0,1))))
                rows.append(dict(atom_ids=ctx.ids(ring),residue_id=resid,receptor_serials=[atoms[a]['serial'] for a in names],
                                 center_distance_angstrom=distance,normal_angle_degrees=angle,
                                 lateral_offset_angstrom=float(np.linalg.norm(delta-np.dot(delta,n)*n)),
                                 candidate_geometry='parallel' if angle<=30 else 'T_shaped' if angle>=60 else 'oblique'))
    return result({'candidate_count':len(rows),'candidates':rows},units={'distance':'angstrom','angle':'degree'},method='Aromatic-ring center, plane-normal angle and lateral displacement',
                  thresholds={'center_distance_angstrom':cutoff},notes=['Geometric candidates only; no interaction energy or mandatory contact inferred.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('aromatic_contacts')
