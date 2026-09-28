"""Numerical validation and copy-trial feedback; not a final-molecule assessment."""
import numpy as np
import torch
from molsteer.common import observation


def gradient_check(fn, coords, epsilon=1e-6):
    x = coords.detach().clone().double().requires_grad_(True)
    analytic = torch.autograd.grad(fn(x), x)[0]
    numeric = torch.zeros_like(x)
    for i in range(x.numel()):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus.reshape(-1)[i] += epsilon
        minus.reshape(-1)[i] -= epsilon
        numeric.reshape(-1)[i] = (fn(plus)-fn(minus))/(2*epsilon)
    error = float((numeric-analytic).abs().max())
    passed = bool(torch.allclose(numeric, analytic, rtol=1e-4, atol=1e-7))
    return dict(passed=passed, maximum_absolute_error=error, finite_difference_epsilon_angstrom=epsilon,
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
