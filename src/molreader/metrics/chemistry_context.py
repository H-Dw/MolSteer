"""Slot-preserving atom, edge and categorical context; no invented defect flags."""
from itertools import combinations
import numpy as np
from ..core import result


def snapshot(ctx):
    atoms=[]
    for i,slot in enumerate(ctx.atom_ids):
        row=dict(atom_id=int(slot),element=ctx.atoms[i],formal_charge=ctx.charges[i],
                 declared_bond_order_sum=float(ctx.orders[i].sum()))
        if ctx.mol is not None:
            a=ctx.mol.GetAtomWithIdx(i)
            row.update(implicit_hydrogens=a.GetNumImplicitHs(),total_valence=a.GetTotalValence(),
                       hybridization=str(a.GetHybridization()))
        if ctx.probs:
            for key,tokens in [('atomics','atomic_tokens'),('charges','charge_tokens')]:
                p=ctx.probs[key][i]
                row[key+'_top3']=[dict(category=ctx.config[tokens][int(k)],probability=float(p[k])) for k in np.argsort(p)[::-1][:3]]
        atoms.append(row)
    bonds=[]
    for i,j in combinations(range(len(ctx.atom_ids)),2):
        row=dict(atom_ids=ctx.ids([i,j]),bond_order=float(ctx.orders[i,j]),
                 distance_angstrom=float(np.linalg.norm(ctx.coords[i]-ctx.coords[j])))
        if ctx.probs:
            p=ctx.probs['bonds'][i,j]
            row['top3']=[dict(category=ctx.config['bond_orders'][int(k)],probability=float(p[k])) for k in np.argsort(p)[::-1][:3]]
        bonds.append(row)
    return dict(atoms=atoms,bonds=bonds)


def compute(ctx):
    data=snapshot(ctx)
    p=ctx.previous
    if p is not None and p.view==ctx.view and all(p.identity[k]==ctx.identity[k] for k in ('target_id','ligand_id')) and p.identity['stage_t']<ctx.identity['stage_t'] and np.array_equal(p.atom_ids,ctx.atom_ids):
        data['previous']=dict(identity=p.identity,source_hashes={k:v['sha256'] for k,v in p.sources.items()},**snapshot(p))
    return result(data,method='Original tensor slots, declared graph, RDKit implicit-H assumptions and top-three model categories',
                  notes=['Slot identity does not imply the same element, bond role or chemical motif over time. Category probabilities are not calibrated defect probabilities.'])


if __name__=='__main__':
    from ..cli import metric_main
    metric_main('chemistry_context')
