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


def probe_live_scalar_response(adapter, packet, biology, drafts, strategy):
    """Optional read-only component pullbacks into one native coordinate tensor."""
    required = ('predict', 'endpoint', 'world_state_coordinates', 'curr', 'index')
    if adapter is None or not all(hasattr(adapter, key) for key in required):
        return dict(status='not_run', scope='live_current_coordinates', reason='No live sampling adapter available')
    try:
        if strategy.get('mode') != 'scalar_potential':
            return dict(status='not_run', reason='A scalar candidate is required')
        evaluator = ExpertEvaluator(dict(mathematical_design=dict(directions=list(drafts.values()), strategy=strategy),
                                         biology_plan=biology), packet)
        x = adapter.curr['coords'].detach().clone().requires_grad_(True)
        pred, _ = probe_predict(adapter, coordinates=x)
        endpoint = adapter.endpoint(pred)
        coordinates = dict(prediction=endpoint['coords'], state=adapter.world_state_coordinates(x))
        molecules, elements = {}, {}
        if evaluator.dynamic_views or evaluator.mmff:
            from molreader.io import load_config
            from .chemistry import decode_endpoint
            vocabulary = load_config()
            for view in evaluator.dynamic_views | set(evaluator.mmff):
                current = endpoint if view == 'prediction' else dict(coords=coordinates['state'],
                    **{k:adapter.curr[k][adapter.index] for k in ('atomics', 'charges', 'bonds')})
                elements[view] = [vocabulary['atomic_tokens'][i] for i in current['atomics'].argmax(-1).tolist()]
                molecules[view] = decode_endpoint(current, vocabulary)
        values = evaluator.components(coordinates, molecules=molecules, elements=elements)
        objectives = evaluator.objectives(values)
        mask = torch.zeros_like(adapter.curr['mask'], dtype=torch.bool)
        editable = adapter.config.get('editable_atom_ids')
        if editable is None:
            mask[adapter.index] = adapter.curr['mask'][adapter.index].bool()
        else:
            mask[adapter.index, editable] = True
        mask[adapter.index, adapter.config.get('fixed_atom_ids', [])] = False
        total = -aggregate_objectives(objectives, strategy)
        grad, = torch.autograd.grad(total + x.sum()*0, x, retain_graph=True)
        grad = grad * mask.unsqueeze(-1)
        rows, gradients = [], []
        for key, value in objectives.items():
            component, = torch.autograd.grad(value + x.sum()*0, x, retain_graph=True)
            component = component * mask.unsqueeze(-1)
            gradients.append(component)
            rows.append(dict(direction_id=key, value=float(value.detach()),
                rank_coefficient=strategy.get('priority_weights', {}).get(key, 1.), gradient_norm=float(component.norm()),
                derivative_along_scalar_reward=float((component*grad).sum())))
        flat = torch.stack(gradients).reshape(len(gradients), -1)
        norm = flat.norm(dim=-1)
        cosine = (flat @ flat.T) / (norm[:, None]*norm[None, :]).clamp(min=1e-30)
        return dict(status='measured', scope='live_current_coordinates', components=rows,
            gradient_cosines=cosine.detach().tolist(), scalar_gradient_norm=float(grad.norm()),
            reference_bindings=evaluator.reference_diagnostics, terminal_benefit='not_run')
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        return dict(status='unavailable', scope='live_current_coordinates', reason=str(exc), blocking=False)


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
                raise ValueError('Reward component unavailable: '+direction['direction_id']+': '+str(exc)) from exc
            finally:
                if bindings:
                    self.reference_diagnostics[direction['direction_id']] = bindings
            values[direction['direction_id']] = evaluate_expression(direction['expression'], local)
        return values

    def objectives(self, values):
        expected = {d['direction_id'] for d in self.directions if self.roles[d['direction_id']] == 'optimize'}
        if expected - set(values):
            raise ValueError('Reward components missing: '+', '.join(sorted(expected-set(values))))
        return {k:v for k,v in values.items() if self.roles[k]=='optimize'}

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
    """Derivative and response measurements on copies; no runtime acceptance policy."""
    from molsteer.molmonitor.checks import gradient_check
    evaluator = ExpertEvaluator(spec, packet)
    if evaluator.strategy['mode'] != 'scalar_potential' or any(r == 'constraint' for r in evaluator.roles.values()):
        raise ValueError('Legacy controller or proposal constraints require scalar redesign')
    views = sorted({o['view'] for d in evaluator.directions for o in d['observables']})
    snapshots = packet['steering']['coordinate_snapshots']
    lengths = [len(snapshots[v]['atom_ids']) for v in views]
    original = torch.cat([torch.tensor(snapshots[v]['coords_angstrom'], dtype=torch.float64) for v in views])
    declared = spec['model_dynamics']['editable_atom_ids']
    mask = torch.tensor([declared is None or a in declared for v in views for a in snapshots[v]['atom_ids']],
                        dtype=original.dtype)[:, None].expand_as(original)
    def components(x):
        return evaluator.components(dict(zip(views, torch.split(x, lengths))))
    initial = components(original)
    checks = {key: gradient_check(lambda x, k=key: components(x)[k], original) for key in initial}
    variable = original.clone().requires_grad_(True)
    values = components(variable)
    objectives = evaluator.objectives(values)
    potential = aggregate_objectives(objectives, evaluator.strategy)
    gradient, = torch.autograd.grad(potential + variable.sum()*0, variable, retain_graph=True)
    rows, gradients = [], []
    weights = evaluator.strategy.get('priority_weights', {})
    by_id = {d['direction_id']: d for d in evaluator.directions}
    for key, value in objectives.items():
        local, = torch.autograd.grad(value + variable.sum()*0, variable, retain_graph=True)
        local = local * mask
        gradients.append(local)
        rows.append(dict(direction_id=key, value=float(value.detach()), rank_coefficient=weights.get(key, 1.),
            gradient_norm=float(local.norm()), directional_derivative=float((local*(-gradient*mask)).sum()),
            physical_scales=by_id[key].get('reference_parameters', [])))
    flat = torch.stack(gradients).reshape(len(gradients), -1)
    norms = flat.norm(dim=-1)
    cosines = (flat @ flat.T) / (norms[:, None]*norms[None, :]).clamp(min=1e-30)
    # This small displacement is a declared numerical response probe, never a sampler step.
    probe_step = 1e-4
    probe = original - probe_step * strength * gradient.detach() * mask
    after = components(probe)
    return dict(numerical_gradient=dict(passed=all(c['passed'] for c in checks.values()), components=checks),
        fixed_atoms_unchanged=bool(torch.equal(probe[mask == 0], original[mask == 0])),
        input_snapshot_unchanged=all(digest(snapshots[v]['coords_angstrom']) == spec['coordinate_hashes'][v] for v in views),
        penalty_before=float(potential.detach()), penalty_after=float(aggregate_objectives(evaluator.objectives(after), evaluator.strategy)),
        components_before={k:float(v) for k,v in initial.items()}, components_after={k:float(v) for k,v in after.items()},
        component_response=rows, gradient_cosines=cosines.detach().tolist(), scalar_gradient_norm=float((gradient*mask).norm()),
        probe_step=probe_step, probe_weight=strength, live_gradient='not_run', full_sampler_ablation='not_run',
        scope='Coordinate-copy response in separate declared views; same-view gradients are comparable. Mixed-view cosines are not a live shared pullback. No terminal benefit established.')
