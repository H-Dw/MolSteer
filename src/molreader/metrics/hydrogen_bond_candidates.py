from ._shared import receptor_distances,receptor_location
from ..core import result,need_mol,features

def compute(ctx):
    mol=need_mol(ctx);protein,d=receptor_distances(ctx);f=features(mol)
    polar={i for family in ('Donor','Acceptor') for group in f.get(family,[]) for i in group}
    cutoff=ctx.config['thresholds']['polar_distance_angstrom'];rows=[]
    for i in sorted(polar):
        for j,a in enumerate(protein):
            if a['record_type']=='ATOM' and a['element'] in ('N','O','S') and d[i,j]<=cutoff:
                rows.append(dict(atom_ids=ctx.ids([i]),distance_angstrom=float(d[i,j]),**receptor_location(a)))
    return result({'candidate_count':len(rows),'candidates':rows},status='partial',units={'distance':'angstrom'},
                  method='Ligand chemically typed donor/acceptor to protein N/O/S distance candidates',thresholds={'distance_angstrom':cutoff},
                  notes=['Protein donor/acceptor typing, protonation and D-H-A angle are not verified. These are polar candidates, not confirmed hydrogen bonds. Absence is not a defect.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('hydrogen_bond_candidates')
