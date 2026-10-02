"""Numerical validation and copy-trial feedback; not a final-molecule assessment."""
import numpy as np
import torch
from molsteer.common import observation


def gradient_check(fn, coords, epsilon=1e-6):
    """Verify the derivative by refinement, including both one-sided limits.

    A squared hinge is differentiable at its boundary, but central differences
    there have O(h) truncation error. Keep the tolerances fixed and require two
    consecutive passing scales; one-sided limits still reject a linear kink.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Finite-difference epsilon must be positive and finite')
    x = coords.detach().clone().double().requires_grad_(True)
    value = fn(x)
    analytic = torch.autograd.grad(value, x)[0]
    base = value.detach()
    refinements = []
    consecutive = 0
    for level in range(5):
        step = epsilon / (4**level)
        forward, backward = torch.zeros_like(x), torch.zeros_like(x)
        with torch.no_grad():
            for i in range(x.numel()):
                plus, minus = x.detach().clone(), x.detach().clone()
                plus.reshape(-1)[i] += step
                minus.reshape(-1)[i] -= step
                forward.reshape(-1)[i] = (fn(plus)-base)/step
                backward.reshape(-1)[i] = (base-fn(minus))/step
        numeric = (forward+backward)/2
        error = float((numeric-analytic).abs().max())
        one_sided_error = max(float((side-analytic).abs().max()) for side in (forward,backward))
        passed_scale = bool(torch.isfinite(analytic).all() and all(
            torch.allclose(side, analytic, rtol=1e-4, atol=1e-7)
            for side in (numeric,forward,backward)))
        refinements.append(dict(epsilon_angstrom=step, maximum_absolute_error=error,
                                maximum_one_sided_error=one_sided_error, passed=passed_scale))
        consecutive = consecutive+1 if passed_scale else 0
        if consecutive >= 2:
            break
    passed = consecutive >= 2
    return dict(passed=passed, maximum_absolute_error=error, finite_difference_epsilon_angstrom=step,
                initial_epsilon_angstrom=epsilon, maximum_one_sided_error=one_sided_error,
                refinement_checks=refinements, tolerances=dict(rtol=1e-4,atol=1e-7),
                analytic_gradient_l2=float(analytic.norm()), checked_coordinates=x.numel(),
                scope='Coordinate derivative on a detached copy; no generator Jacobian tested')


def screen_copy(packet, view, coords):
    from rdkit import Chem
    snapshot = packet['steering']['coordinate_snapshots'][view]
    slot = {a:i for i,a in enumerate(snapshot['atom_ids'])}
    x = np.asarray(coords)
    bond = observation(packet, 'bond_lengths', view)
    violations = []
    for row in (bond['values'].get('bonds', []) if bond else []):
        low, high = row.get('lower_angstrom'), row.get('upper_angstrom')
        if low is None or high is None:
            continue
        a,b = row['atom_ids']; d = float(np.linalg.norm(x[slot[a]]-x[slot[b]]))
        if d < low or d > high:
            violations.append(dict(atom_ids=[a,b], distance_angstrom=d))
    context = observation(packet, 'chemistry_context', view)
    cm = observation(packet, 'protein_clashes', view)
    threshold = cm['thresholds']['distance_over_vdw_sum_below'] if cm else None
    clash, checked = [], 0
    if context and threshold:
        pt = Chem.GetPeriodicTable()
        for a in context['values']['atoms']:
            if not a['element'] or a['element']=='H':
                continue
            for p in packet['steering']['receptor_atoms']:
                if p['record_type']!='ATOM':
                    continue
                checked += 1
                d = float(np.linalg.norm(x[slot[a['atom_id']]]-np.asarray(p['coords'])))
                minimum = threshold*(pt.GetRvdw(pt.GetAtomicNumber(a['element']))+pt.GetRvdw(p['atomic_number']))
                if d < minimum:
                    clash.append(dict(atom_id=a['atom_id'], receptor_serial=p['serial'], penetration_angstrom=minimum-d))
    return dict(bond_window_violations=violations, protein_clashes=clash, checked_protein_pairs=checked,
                scope='Frozen-graph bond windows and heavy-atom protein overlap; not full post-steering validation')
