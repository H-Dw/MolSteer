from ._shared import receptor_distances,receptor_location
from ..core import result,need_mol,features

def compute(ctx):
    mol=need_mol(ctx);protein,d=receptor_distances(ctx);f=features(mol)
    hydro={i for family in ('Hydrophobe','LumpedHydrophobe') for group in f.get(family,[]) for i in group}
    cutoff=ctx.config['thresholds']['hydrophobic_distance_angstrom'];rows=[]
    for i in sorted(hydro):
        for j,a in enumerate(protein):
            if a['record_type']=='ATOM' and a['element']=='C' and a['atom_name']!='C' and d[i,j]<=cutoff:
                rows.append(dict(atom_ids=ctx.ids([i]),distance_angstrom=float(d[i,j]),**receptor_location(a)))
    return result({'candidate_count':len(rows),'candidates':rows},status='partial',units={'distance':'angstrom'},
                  method='RDKit ligand hydrophobes to protein non-carbonyl carbon distance screen',thresholds={'distance_angstrom':cutoff},
                  notes=['Protein chemical typing is approximate without an explicit prepared graph. This count is not an affinity score.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('hydrophobic_contacts')
