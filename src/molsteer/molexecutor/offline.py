"""Bounded projected descent on a copy, with explicit demo-only editable atoms."""
import torch
from molsteer.common import digest
from molsteer.molthinker.planner import validate_spec
from molsteer.molmonitor.checks import gradient_check, screen_copy
from .energies import energy, observable, term_energy
from molsteer.molthinker.composition import objective_value


def run_offline_trial(packet, spec, view, demo_movable_atom_ids, iterations=30,
                      learning_rate=.25, max_step_angstrom=.03, trust_radius_angstrom=.20):
    validate_spec(spec, packet)
    if iterations < 1 or iterations > 1000 or min(learning_rate,max_step_angstrom,trust_radius_angstrom)<=0:
        raise ValueError('Invalid bounded trial settings')
    snap = packet['steering']['coordinate_snapshots'][view]
    if digest(snap['coords_angstrom']) != spec['coordinate_hashes'][view]:
        raise ValueError('Snapshot changed after reward derivation')
    if packet['steering']['graph_signatures'] != spec['graph_signatures']:
        raise ValueError('Chemical graph changed; rederive the reward')
    slots = snap['atom_ids']; index = {a:i for i,a in enumerate(slots)}
    if not demo_movable_atom_ids or not set(demo_movable_atom_ids) <= set(slots):
        raise ValueError('Explicit valid demo mobility mask required')
    terms = [t for t in spec['terms'] if t['view']==view]
    if not terms:
        raise ValueError('No reward terms for this representation')
    original = torch.tensor(snap['coords_angstrom'], dtype=torch.float64)
    coords = original.clone()
    mask = torch.tensor([a in demo_movable_atom_ids for a in slots],dtype=torch.float64)[:,None]
    fn = lambda x: energy(terms,x,slots)
    numerical = gradient_check(fn, coords)
    if not numerical['passed']:
        raise ValueError('Finite-difference validation failed')
    before_screen = screen_copy(packet,view,original.tolist())
    trace = [float(fn(coords))]
    for _ in range(iterations):
        current = coords.detach().requires_grad_(True)
        grad = torch.autograd.grad(fn(current),current)[0]*mask
        delta = -learning_rate*grad
        delta *= torch.clamp(max_step_angstrom/torch.clamp(delta.norm(dim=1,keepdim=True),min=1e-30),max=1.)
        accepted = False
        for _ in range(12):
            offset = (coords+delta-original)*mask
            offset *= torch.clamp(trust_radius_angstrom/torch.clamp(offset.norm(dim=1,keepdim=True),min=1e-30),max=1.)
            candidate = original+offset
            value = float(fn(candidate))
            if torch.isfinite(candidate).all() and value <= trace[-1]+1e-14:
                coords = candidate.detach();trace.append(value);accepted=True;break
            delta *= .5
        if not accepted or trace[-1] < 1e-12:
            break
    after_screen = screen_copy(packet,view,coords.tolist())
    before_bonds={tuple(r['atom_ids']) for r in before_screen['bond_window_violations']}
    after_bonds={tuple(r['atom_ids']) for r in after_screen['bond_window_violations']}
    before_clashes={(r['atom_id'],r['receptor_serial']) for r in before_screen['protein_clashes']}
    after_clashes={(r['atom_id'],r['receptor_serial']) for r in after_screen['protein_clashes']}
    measurement = lambda t,x: dict(value=float(observable(t,x,index)), energy=float(term_energy(t,x,index)))
    records = [dict(term_id=t['term_id'], atom_ids=t['atom_ids'], unit=t['unit'], before=measurement(t,original), after=measurement(t,coords)) for t in terms]
    fixed = mask[:,0] == 0
    return dict(kind='ExecutionMonitor', packet_id=packet['packet_id'], reward_id=spec['reward_id'],view=view,
        mode='offline_frozen_graph_coordinate_copy', demo_movable_atom_ids=sorted(demo_movable_atom_ids),
        mobility_declaration='Demonstration only; does not establish editable atoms in the generator',
        settings=dict(iterations=iterations,learning_rate=learning_rate,max_step_angstrom=max_step_angstrom,trust_radius_angstrom=trust_radius_angstrom),
        numerical_gradient=numerical, penalty_before=trace[0],penalty_after=trace[-1],reward_before=-trace[0],reward_after=-trace[-1],
        delta_E=trace[-1]-trace[0], delta_norm_angstrom=float((coords-original).norm()),
        max_atom_displacement_angstrom=float((coords-original).norm(dim=1).max()),
        fixed_atoms_unchanged=bool(torch.equal(coords[fixed],original[fixed])),
        input_snapshot_unchanged=digest(snap['coords_angstrom'])==spec['coordinate_hashes'][view],
        monotone_penalty_descent=all(b<=a+1e-14 for a,b in zip(trace,trace[1:])),trace=trace,
        terms=records, before_screen=before_screen,after_screen=after_screen,
        new_bond_window_violations=len(after_bonds-before_bonds),resolved_bond_window_violations=len(before_bonds-after_bonds),
        new_protein_clashes=len(after_clashes-before_clashes),resolved_protein_clashes=len(before_clashes-after_clashes),
        trial_coordinates_angstrom=coords.tolist(), original_atom_ids=slots,
        rebound=None, population_ESS=None, full_sampler_ablation='not_run',
        conclusion='Numerical feasibility only; not evidence of generator control, stable chemistry or improved final binding')


def run_offline_design_trial(packet, spec, iterations=3, learning_rate=.25,
                             max_step_angstrom=.03, trust_radius_angstrom=.20):
    """Check the selected tree jointly across its declared coordinate views on copies."""
    validate_spec(spec, packet)
    if 'design' not in spec or not 1<=iterations<=20 or min(learning_rate,max_step_angstrom,trust_radius_angstrom)<=0:
        raise ValueError('Invalid creative copy trial')
    views=[v for v in ('prediction','state') if any(t['view']==v for t in spec['terms'])]
    snapshots={v:packet['steering']['coordinate_snapshots'][v] for v in views}
    if any(digest(snapshots[v]['coords_angstrom'])!=spec['coordinate_hashes'][v] for v in views):
        raise ValueError('Snapshot changed after reward design')
    if packet['steering']['graph_signatures']!=spec['graph_signatures']:
        raise ValueError('Chemical graph changed; rederive the reward')
    lengths={v:len(snapshots[v]['atom_ids']) for v in views}
    starts={v:sum(lengths[w] for w in views[:i]) for i,v in enumerate(views)}
    indices={v:{a:i for i,a in enumerate(snapshots[v]['atom_ids'])} for v in views}
    original=torch.cat([torch.tensor(snapshots[v]['coords_angstrom'],dtype=torch.float64) for v in views])
    selected={v:{a for t in spec['terms'] if t['view']==v for a in t['atom_ids']} for v in views}
    mask=torch.cat([torch.tensor([a in selected[v] for a in snapshots[v]['atom_ids']],dtype=torch.float64)
                    for v in views])[:,None]

    def pieces(coords):
        return {v:coords[starts[v]:starts[v]+lengths[v]] for v in views}

    def components(coords):
        by_view=pieces(coords)
        return {t['term_id']:term_energy(t,by_view[t['view']],indices[t['view']]) for t in spec['terms']}

    fn=lambda coords:objective_value(spec['design']['objective_tree'],components(coords))
    numerical=gradient_check(fn,original)
    if not numerical['passed']:
        raise ValueError('Creative objective finite-difference validation failed')
    coords=original.clone()
    trace=[float(fn(coords))]
    for _ in range(iterations):
        current=coords.detach().requires_grad_(True)
        gradient=torch.autograd.grad(fn(current),current)[0]*mask
        if not torch.isfinite(gradient).all():
            raise ValueError('Creative objective has nonfinite gradients')
        delta=-learning_rate*gradient
        delta*=torch.clamp(max_step_angstrom/delta.norm(dim=1,keepdim=True).clamp(min=1e-30),max=1.)
        accepted=False
        for _ in range(12):
            offset=(coords+delta-original)*mask
            offset*=torch.clamp(trust_radius_angstrom/offset.norm(dim=1,keepdim=True).clamp(min=1e-30),max=1.)
            candidate=original+offset
            value=float(fn(candidate))
            if torch.isfinite(candidate).all() and value<=trace[-1]+1e-14:
                coords=candidate.detach();trace.append(value);accepted=True;break
            delta*=.5
        if not accepted or trace[-1]<1e-12:
            break
    before=components(original)
    after=components(coords)
    term_records=[dict(term_id=t['term_id'],view=t['view'],unit=t['unit'],
                       value_before=float(observable(t,pieces(original)[t['view']],indices[t['view']])),
                       value_after=float(observable(t,pieces(coords)[t['view']],indices[t['view']])),
                       penalty_before=float(before[t['term_id']]),penalty_after=float(after[t['term_id']]))
                  for t in spec['terms']]
    fixed=mask[:,0]==0
    return dict(kind='ExecutionMonitor',packet_id=packet['packet_id'],reward_id=spec['reward_id'],
        view='joint_declared_views',mode='offline_frozen_graph_coordinate_copy',
        numerical_gradient=numerical,penalty_before=trace[0],penalty_after=trace[-1],
        reward_before=-trace[0],reward_after=-trace[-1],trace=trace,terms=term_records,
        fixed_atoms_unchanged=bool(torch.equal(coords[fixed],original[fixed])),
        input_snapshot_unchanged=all(digest(snapshots[v]['coords_angstrom'])==spec['coordinate_hashes'][v] for v in views),
        monotone_penalty_descent=all(b<=a+1e-14 for a,b in zip(trace,trace[1:])),
        screen_before={v:screen_copy(packet,v,pieces(original)[v].tolist()) for v in views},
        screen_after={v:screen_copy(packet,v,pieces(coords)[v].tolist()) for v in views},
        limitation='Independent view-coordinate copy test; live endpoint Jacobian and full FLOWR continuation not tested')


def execute_live(config=None):
    if config is None:
        raise NotImplementedError('Live execution requires an explicit generator adapter, editable masks, control budget and validated endpoint-to-state derivative')
    from .runner import run
    return run(config)
