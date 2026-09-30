"""Graph changes are review events, never a requirement to preserve identity."""
from copy import deepcopy
from molsteer.common import digest, observation


def packet_graphs(packet):
    graphs={}
    for view in ('state','prediction'):
        context=observation(packet,'chemistry_context',view)
        if not context:continue
        values=context['values']
        atoms=[{key:a.get(key) for key in ('atom_id','element','formal_charge')} for a in values['atoms']]
        graphs[view]=dict(atoms=sorted(atoms,key=lambda a:a['atom_id']),
            bonds=sorted([list(b['atom_ids'])+[b['bond_order']] for b in values['bonds']]))
    return graphs


def tensor_graph(pred,vocabulary):
    atomic=pred['atomics'].detach().argmax(-1).cpu().tolist()
    charge=pred['charges'].detach().argmax(-1).cpu().tolist()
    classes=pred['bonds'].detach().argmax(-1).cpu().tolist()
    return dict(atoms=[dict(atom_id=i,element=vocabulary['atomic_tokens'][a],
                           formal_charge=vocabulary['charge_tokens'][q]) for i,(a,q) in enumerate(zip(atomic,charge))],
                # Preserve invalid/asymmetric categorical hypotheses for detection too.
                bonds=[[i,j,vocabulary['bond_orders'][b]] for i,row in enumerate(classes)
                       for j,b in enumerate(row) if i<=j and vocabulary['bond_orders'][b]!=0],
                directed_bond_classes=classes)


def graph_changes(previous,current):
    changes={}
    for view in sorted(set(previous)|set(current)):
        old,new=previous.get(view),current.get(view)
        if old==new:continue
        old_atoms={a['atom_id']:a for a in old['atoms']} if old else {}
        new_atoms={a['atom_id']:a for a in new['atoms']} if new else {}
        changes[view]=dict(previous_graph_id=digest(old),current_graph_id=digest(new),
            slot_mapping_changed=set(old_atoms)!=set(new_atoms),
            atom_changes=[dict(atom_id=i,previous=old_atoms.get(i),current=new_atoms.get(i))
                          for i in sorted(set(old_atoms)|set(new_atoms)) if old_atoms.get(i)!=new_atoms.get(i)],
            bonds_changed=bool(not old or not new or old.get('bonds')!=new.get('bonds') or
                old.get('directed_bond_classes')!=new.get('directed_bond_classes')))
    return changes


def review_event(changes,*,step,parent_program_id=None):
    event=dict(kind='ChemicalGraphChange',route='reader',action='review_chemical_roles',step=step,
        changes=deepcopy(changes),parent_program_id=parent_program_id,
        workflow=['MolMonitor','MolReader','MolThinker','MolExecutor'],
        slot_policy='Preserve native tensor IDs; reassign chemical roles and reward support, never reorder by element',
        review_questions=[
            'Which reward terms remain applicable to the new elements, formal charges and bonds?',
            'Which references, chemical roles or expression constants need rederivation?',
            'Keep the mathematical form if still valid; justify any parameter, support or function change.',
            'Validate the revised program and live derivatives before extra guidance; preserve native sampler state and budget.'])
    event['event_id']='gc_'+digest(event)[:24]
    return event


def packet_change_event(previous,current,*,step,parent_program_id=None):
    # A time/coordinate update alone must not activate the chemical review path.
    for key in ('target_id','ligand_id'):
        if previous['identity'].get(key)!=current['identity'].get(key):
            raise ValueError('Graph review cannot switch target or ligand')
    changes=graph_changes(packet_graphs(previous),packet_graphs(current))
    return review_event(changes,step=step,parent_program_id=parent_program_id) if changes else None
