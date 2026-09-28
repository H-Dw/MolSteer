import numpy as np
from ..core import result,Unavailable

def compute(ctx):
    p=ctx.previous
    if p is None:raise Unavailable('No earlier stage supplied')
    if ctx.identity['target_id']!=p.identity['target_id'] or ctx.identity['ligand_id']!=p.identity['ligand_id'] or ctx.view!=p.view:
        raise Unavailable('History must be same target, ligand and representation')
    if p.identity['stage_t']>=ctx.identity['stage_t']:raise Unavailable('History must precede current t')
    if not np.array_equal(ctx.atom_ids,p.atom_ids):raise Unavailable('Stable tensor-slot identity unavailable')
    if not ctx.world_valid or not p.world_valid:raise Unavailable('History requires verified world coordinates')
    displacement=np.linalg.norm(ctx.coords-p.coords,axis=-1)
    i,j=np.triu_indices(len(ctx.atoms),1)
    changes=ctx.orders[i,j]!=p.orders[i,j]
    return result({'previous_t':p.identity['stage_t'],'current_t':ctx.identity['stage_t'],
                   'world_frame_rms_displacement_angstrom':float(np.sqrt(np.mean(displacement**2))),
                   'atom_type_change_count':sum(a!=b for a,b in zip(ctx.atoms,p.atoms)),
                   'formal_charge_change_count':sum(a!=b for a,b in zip(ctx.charges,p.charges)),
                   'bond_class_change_count':int(changes.sum()),
                   'atom_changes':[{'atom_id':int(ctx.atom_ids[k]),'previous_element':p.atoms[k],'element':ctx.atoms[k],
                                    'previous_charge':p.charges[k],'formal_charge':ctx.charges[k]} for k in range(len(ctx.atoms)) if ctx.atoms[k]!=p.atoms[k] or ctx.charges[k]!=p.charges[k]],
                   'bond_changes':[{'atom_ids':ctx.ids([int(a),int(b)]),'previous_order':float(p.orders[a,b]),'bond_order':float(ctx.orders[a,b])} for a,b in zip(i[changes],j[changes])],
                   'per_atom_displacement':[{'atom_id':int(i),'displacement_angstrom':float(v)} for i,v in zip(ctx.atom_ids,displacement)]},
                  units={'displacement':'angstrom'},method='Stable tensor-slot differences without alignment',
                  notes=['Generation time is not physical time. Slot correspondence does not imply immutable element identity. Changes do not establish causal intervention effects.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('trajectory_change')
