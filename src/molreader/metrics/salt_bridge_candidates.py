from ._shared import receptor_distances,receptor_location
from ..core import result

def compute(ctx):
    protein,d=receptor_distances(ctx);rows=[]
    negative={'ASP':{'OD1','OD2'},'GLU':{'OE1','OE2'}}
    positive={'LYS':{'NZ'},'ARG':{'NE','NH1','NH2'}}
    cutoff=ctx.config['thresholds']['salt_distance_angstrom']
    for i,q in enumerate(ctx.charges):
        if q is None or q==0:continue
        for j,a in enumerate(protein):
            assumed=-1 if a['atom_name'] in negative.get(a['residue_name'],set()) else 1 if a['atom_name'] in positive.get(a['residue_name'],set()) else 0
            if q*assumed<0 and d[i,j]<=cutoff:
                rows.append(dict(atom_ids=ctx.ids([i]),ligand_formal_charge=q,assumed_residue_charge_sign=assumed,distance_angstrom=float(d[i,j]),**receptor_location(a)))
    return result({'candidate_atom_pair_count':len(rows),'candidates':rows},status='partial',units={'distance':'angstrom'},
                  method='Opposite formal-charge/residue-sign distance candidates',thresholds={'distance_angstrom':cutoff},
                  notes=['Assumes ASP/GLU negative and LYS/ARG positive; HIS and termini unassigned. Atom pairs are not unique salt bridges. No pH/protonation verification.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('salt_bridge_candidates')
