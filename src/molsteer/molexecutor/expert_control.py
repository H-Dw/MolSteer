"""Shared expert expression evaluation, coordinate-copy trials and live pullbacks."""
from copy import deepcopy
import numpy as np
import torch
from molsteer.common import digest, observation
from molsteer.molthinker.expressions import observable_value, evaluate_expression, aggregate_objectives
from molsteer.agents.optimization import conflict_weights, priority_descent


def probe_predict(adapter, **kwargs):
    """Probe from the native RNG/conditioning state and restore it even on failure.

    Each probe sees the same production random stream. Model parameters and
    static encodings are read-only under the SamplingAdapter contract.
    """
    from .flowr import snapshot_rng, restore_rng, tree_map
    saved_rng=snapshot_rng()
    saved={name:tree_map(lambda x:x.clone(),getattr(adapter,name))
           for name in ('curr','cond') if hasattr(adapter,name)}
    try:
        return adapter.predict(**kwargs)
    finally:
        for name,value in saved.items():setattr(adapter,name,value)
        restore_rng(saved_rng)


def packet_molecule(packet, view):
    from molreader.io import decode
    context = observation(packet, 'chemistry_context', view)
    if not context:
        raise ValueError('Bound chemical graph is unavailable')
    snap = packet['steering']['coordinate_snapshots'][view]
    ids = snap['atom_ids']; indices = {a:i for i,a in enumerate(ids)}
    atoms = {a['atom_id']:a for a in context['values']['atoms']}
    orders = np.zeros((len(ids), len(ids)))
    for bond in context['values']['bonds']:
        i,j = [indices[a] for a in bond['atom_ids']]
        orders[i,j] = orders[j,i] = bond['bond_order']
    _, mol, error = decode(np.array(snap['coords_angstrom']), [atoms[a]['element'] for a in ids],
                           [atoms[a]['formal_charge'] for a in ids], orders)
    if mol is None:
        raise ValueError('Bound MMFF graph cannot be decoded')
    return mol


class ExpertEvaluator:
    def __init__(self, spec, packet):
        self.spec, self.packet = spec, packet
        self.directions = [d for d in spec['mathematical_design']['directions'] if d['status']=='executable']
        self.strategy = spec['mathematical_design']['strategy']
        self.roles = {d['direction_id']:d['disposition'] for d in spec['biology_plan']['directions']}
        self.dependencies = {d['direction_id']:d['preservation_conditions'] for d in spec['biology_plan']['directions']}
        self.constraint_modes = {d['direction_id']:d.get('constraint_mode', 'absolute') for d in self.directions}
        self.dynamic_views = {o['view'] for d in self.directions for o in d['observables']
                              if o['kind'] in ('bond_length_error', 'bond_angle_error', 'typed_steric_overlap')}
        self.reference_diagnostics = {}; self.unavailable = {}
        self.mmff, self.molecules = {}, {}
        from .mmff_bridge import MMFFStrain
        for direction in self.directions:
            for obs in direction['observables']:
                if obs['kind']=='mmff_strain' and obs['view'] not in self.mmff:
                    view = obs['view']
                    ready = packet['steering']['chemical_readiness'][view]
                    if any(ready.get(k,{}).get('value') is not True for k in
                           ('graph_valid','protonation_validated','mmff_applicability_validated')):
                        raise ValueError('MMFF chemical applicability and protonation must be validated')
                    self.mmff[view], self.molecules[view] = MMFFStrain(), packet_molecule(packet,view)
        for view in self.dynamic_views - set(self.molecules):
            try:
                self.molecules[view] = packet_molecule(packet, view)
            except ValueError:
                self.molecules[view] = None
        self.elements = {}
        for view in self.dynamic_views:
            chemical = observation(packet, 'chemistry_context', view)
            atoms = {a['atom_id']:a['element'] for a in (chemical or {}).get('values', {}).get('atoms', [])}
            self.elements[view] = [atoms.get(a) for a in packet['steering']['coordinate_snapshots'][view]['atom_ids']]

    def components(self, coordinates, molecules=None, elements=None):
        # Offline trials use their bound snapshot. Live callers supply freshly
        # decoded molecules so force-field typing follows the current categories.
        molecules = self.molecules if molecules is None else molecules
        elements = self.elements if elements is None else elements
        from .chemical_references import ChemicalReferenceUnavailable
        self.reference_diagnostics = {}; self.unavailable = {}
        values = {}
        for direction in self.directions:
            local, bindings = {}, {}
            try:
                for obs in direction['observables']:
                    view = obs['view']
                    local[obs['observable_id']] = observable_value(obs, coordinates[view],
                        self.packet['steering']['coordinate_snapshots'][view]['atom_ids'], self.packet,
                        self.mmff.get(view), molecules.get(view), elements.get(view), bindings)
            except ChemicalReferenceUnavailable as exc:
                self.unavailable[direction['direction_id']] = str(exc)
                continue
            finally:
                if bindings:
                    self.reference_diagnostics[direction['direction_id']] = bindings
            values[direction['direction_id']] = evaluate_expression(direction['expression'], local)
        return values

    def objectives(self, values):
        return {k:v for k,v in values.items() if self.roles[k]=='optimize' and
                all(c in values for c in self.dependencies.get(k, []))}

    def constraints(self, values, baseline=None):
        return [k for k,v in values.items() if self.roles[k]=='constraint' and
                float(v.detach()) > (float(baseline[k].detach()) if self.constraint_modes[k] == 'native_nonincrease'
                                     and baseline is not None and k in baseline else 0.) + 1e-10]


def control_direction(objectives, variable, mask, strategy):
    """Pull back each scalar through the same live variable before any comparison."""
    if mask.shape == variable.shape[:-1]: mask = mask.unsqueeze(-1).expand_as(variable)
    if mask.shape != variable.shape or not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Control mask must match the runtime variable')
    grads = []
    if not objectives:
        raise ValueError('No currently applicable optimization mechanism')
    for value in objectives.values():
        grad, = torch.autograd.grad(value, variable, retain_graph=True, allow_unused=False)
        if not torch.isfinite(grad).all():
            raise ValueError('Nonfinite objective derivative')
        grads.append(grad*mask)
    stacked = torch.stack(grads)
    diagnostics = conflict_weights(stacked.detach().double().cpu().numpy(), mask.detach().cpu().numpy())
    diagnostics['direction_ids'] = list(objectives)
    diagnostics['derivative_scope'] = 'actual supplied variable; runtime owns its mapping'
    if strategy['mode']=='common_descent':
        # Satisfied zero-gradient terms are inactive, not a veto on unsatisfied repairs.
        active = [i for i,v in enumerate(objectives.values()) if float(v.detach()) > 1e-10]
        if not active:
            diagnostics['status']='satisfied'
            return torch.zeros_like(variable), stacked.detach(), diagnostics
        if any(float(stacked[i].norm()) <= 1e-12 for i in active):
            diagnostics['status']='unresolved_zero_gradient'
            return torch.zeros_like(variable), stacked.detach(), diagnostics
        arrays = stacked[active].detach().double().cpu().numpy(), mask.detach().cpu().numpy()
        weights = strategy.get('priority_weights')
        solved = (priority_descent(*arrays, [weights.get(list(objectives)[i], 1.) for i in active]) if weights
                  else conflict_weights(*arrays))
        diagnostics['active_solution'] = solved
        diagnostics['active_direction_ids'] = [list(objectives)[i] for i in active]
        diagnostics['status'] = solved['status']
        if solved['status'] != 'candidate_descent':
            return torch.zeros_like(variable), stacked.detach(), diagnostics
        direction = variable.new_tensor(solved['projected_direction'])
    else:
        potential = aggregate_objectives(objectives, strategy)
        gradient, = torch.autograd.grad(potential, variable, retain_graph=True)
        direction = -gradient*mask
        diagnostics['status']='scalar_potential' if float(direction.norm()) > 0 else 'stationary'
    if not torch.isfinite(direction).all():
        raise ValueError('Nonfinite control direction')
    return direction.detach(), stacked.detach(), diagnostics


def check_displacement(delta, gradients, strategy, tolerance=1e-10, scalar_gradient=None):
    values = (gradients*delta).reshape(len(gradients),-1).sum(-1)
    valid = bool(torch.isfinite(delta).all() and torch.isfinite(values).all())
    if strategy['mode']=='common_descent':
        valid = valid and bool((values <= tolerance).all()) and bool((values < -tolerance).any())
    elif scalar_gradient is None:
        raise ValueError('Scalar direction checks require the actual potential gradient')
    else:
        derivative=(scalar_gradient*delta).sum()
        valid=valid and bool(torch.isfinite(derivative)) and float(derivative)<=tolerance
    return valid, values.detach().cpu().tolist()


def run_expert_trial(packet, spec, iterations=3, strength=1.):
    from molsteer.molmonitor.checks import gradient_check
    evaluator = ExpertEvaluator(spec,packet)
    views = sorted({o['view'] for d in evaluator.directions for o in d['observables']})
    snapshots = packet['steering']['coordinate_snapshots']
    lengths = [len(snapshots[v]['atom_ids']) for v in views]
    original = torch.cat([torch.tensor(snapshots[v]['coords_angstrom'],dtype=torch.float64) for v in views])
    supports = {v:{a for d in evaluator.directions for o in d['observables'] if o['view']==v for a in o['atom_ids']} for v in views}
    declared = spec['model_dynamics']['editable_atom_ids']
    mask = torch.tensor([a in supports[v] and (declared is None or a in declared)
                         for v in views for a in snapshots[v]['atom_ids']],dtype=original.dtype)[:,None].expand_as(original)
    def components(x):
        return evaluator.components(dict(zip(views,torch.split(x,lengths))))
    initial = components(original)
    if not evaluator.objectives(initial):
        return {'numerical_gradient':{'passed':False, 'components':{}},
                'unavailable_directions':dict(evaluator.unavailable),
                'reference_bindings':dict(evaluator.reference_diagnostics),
                'fixed_atoms_unchanged':True, 'input_snapshot_unchanged':True,
                'live_gradient':'not_run', 'full_sampler_ablation':'not_run',
                'scope':'No currently applicable optimization mechanism on the coordinate copy',
                'revision_hint':'Inspect missing current chemical references and preservation dependencies; revise the mechanism or retain it as unresolved.'}
    checks = {key:gradient_check(lambda x,k=key:components(x)[k], original) for key in initial}
    if not all(c['passed'] for c in checks.values()):
        return {'numerical_gradient':{'passed':False,'components':checks},
                'fixed_atoms_unchanged':True,'input_snapshot_unchanged':True,
                'live_gradient':'not_run','full_sampler_ablation':'not_run',
                'scope':'Failed coordinate-copy derivative check; no displacement attempted',
                'revision_hint':'Inspect the failed components and their active/stopping regions. A linear hinge exactly at its reference has no classical derivative. Review target feasibility with biology; do not loosen numerical tolerances or claim live validation.'}
    current = original.clone(); rows = []
    for _ in range(iterations):
        variable = current.detach().requires_grad_(True)
        values = components(variable)
        objectives = evaluator.objectives(values)
        direction, gradients, diagnostic = control_direction(objectives,variable,mask,evaluator.strategy)
        delta = .25*strength*direction
        delta *= (.03/delta.norm(dim=-1,keepdim=True).clamp(min=1e-30)).clamp(max=1)
        accepted = False; reasons=[]
        for attempt in range(12):
            candidate = current+delta*(.5**attempt)
            offset = candidate-original
            candidate = original+offset*(.20/offset.norm(dim=-1,keepdim=True).clamp(min=1e-30)).clamp(max=1)
            ok, derivatives = check_displacement(candidate-current, gradients, evaluator.strategy,scalar_gradient=-direction)
            after = components(candidate)
            reasons = evaluator.constraints(after, values)
            if evaluator.strategy['mode']=='common_descent':
                regression = any(float(after[k])>float(values[k])+1e-10 for k in objectives)
            else:
                regression = bool(aggregate_objectives(evaluator.objectives(after),evaluator.strategy) >
                                  aggregate_objectives(objectives,evaluator.strategy)+1e-10)
            if ok and not reasons and not regression:
                current=candidate.detach(); accepted=True; break
        rows.append(dict(accepted=accepted,conflict=diagnostic,post_clip_directional_derivatives=derivatives,
                         constraint_failures=reasons))
        if not accepted or float(direction.norm())==0: break
    final = components(current)
    return {'numerical_gradient':{'passed':True,'components':checks}, 'fixed_atoms_unchanged':bool(torch.equal(current[mask==0],original[mask==0])),
            'input_snapshot_unchanged':all(digest(snapshots[v]['coords_angstrom'])==spec['coordinate_hashes'][v] for v in views),
            'penalty_before':float(aggregate_objectives(evaluator.objectives(initial),evaluator.strategy)),
            'penalty_after':float(aggregate_objectives(evaluator.objectives(final),evaluator.strategy)),
            'components_before':{k:float(v) for k,v in initial.items()},'components_after':{k:float(v) for k,v in final.items()},
            'control_trials':rows,'live_gradient':'not_run','full_sampler_ablation':'not_run',
            'scope':'Independent frozen coordinate views; no claim about live conflicts or terminal outcomes'}
