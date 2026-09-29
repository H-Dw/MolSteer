"""Versioned, dimension-checked scalar expressions and molecular observables.

No strings are evaluated as code. Coordinates are world Angstrom; angular
observables explicitly declare radians or degrees. Each evaluation rechecks
the numerical domain, including geometry singularities.
"""
from copy import deepcopy
import math
import torch

UNITS = {'dimensionless': (0, 0, 0), 'angstrom': (1, 0, 0),
         'angstrom^3': (3, 0, 0), 'radian': (0, 1, 0), 'degree': (0, 1, 0),
         'kcal/mol': (0, 0, 1)}
OBSERVABLES = {
    'distance': (2, 'angstrom'), 'receptor_distance': (1, 'angstrom'),
    'angle': (3, 'radian'), 'anchor_offset': (1, 'angstrom'),
    'direction_alignment': (2, 'dimensionless'), 'dihedral': (4, 'radian'),
    'signed_volume': (4, 'angstrom^3'), 'mmff_strain': (None, 'kcal/mol'),
}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e12


def validate_observables(observables, packet, evidence_ids):
    if not isinstance(observables, list) or not 1 <= len(observables) <= 32:
        raise ValueError('Use 1 to 32 bound observables per direction')
    ids = set()
    evidence_index={e['evidence_id']:(m['view'],e) for m in packet['observations'] for e in m.get('evidence',[])}
    for obs in observables:
        if set(obs) != {'observable_id', 'kind', 'view', 'atom_ids', 'evidence_ids', 'parameters'}:
            raise ValueError('Observable fields disagree with contract')
        name, kind, view = obs['observable_id'], obs['kind'], obs['view']
        if not isinstance(name, str) or not name or name in ids or kind not in OBSERVABLES:
            raise ValueError('Unknown observable kind or duplicate ID')
        ids.add(name)
        if view not in ('state', 'prediction'):
            raise ValueError('Only state and prediction views are executable')
        representation=packet['representations'][view]
        if representation['coordinate_unit']!='angstrom':
            raise ValueError('Observable coordinates must use declared Angstrom units')
        if kind in ('receptor_distance','anchor_offset','direction_alignment') and (
                representation['coordinate_frame']!='receptor_world' or not representation.get('transform',{}).get('verified')):
            raise ValueError('Reference observables require a verified common world frame')
        atoms = obs['atom_ids']
        valid = packet['representations'][view]['original_atom_ids']
        count, _ = OBSERVABLES[kind]
        if (not isinstance(atoms, list) or not atoms or any(type(a) is not int for a in atoms)
                or len(set(atoms)) != len(atoms) or not set(atoms) <= set(valid)
                or count is not None and len(atoms) != count):
            raise ValueError('Observable atom mapping or arity is invalid')
        if not obs['evidence_ids'] or not set(obs['evidence_ids']) <= evidence_ids:
            raise ValueError('Observable must cite its direction diagnostic evidence')
        if any(e not in evidence_index or evidence_index[e][0]!=view for e in obs['evidence_ids']):
            raise ValueError('Observable and evidence must use the same representation')
        bound_atoms={a for e in obs['evidence_ids'] for a in evidence_index[e][1].get('atom_ids',[])}
        if kind!='mmff_strain' and not set(atoms)<=bound_atoms:
            raise ValueError('Observable support is not localized by its evidence')
        params = obs['parameters']
        required = {'receptor_distance': {'receptor_serial', 'residue_id'},
                    'anchor_offset': {'reference', 'origin'},
                    'direction_alignment': {'reference', 'origin'}}.get(kind, set())
        if not isinstance(params, dict) or set(params) != required:
            raise ValueError('Unexpected or missing observable parameters')
        if 'reference' in required:
            ref = params['reference']
            if not isinstance(ref, list) or len(ref) != 3 or not all(finite_number(x) for x in ref):
                raise ValueError('Reference must be a finite three-vector')
            if not isinstance(params['origin'], str) or len(params['origin'].strip()) < 8:
                raise ValueError('Reference provenance is required')
            if kind == 'direction_alignment' and sum(x*x for x in ref) < 1e-16:
                raise ValueError('Reference direction is degenerate')
        if kind == 'receptor_distance':
            receptor_reference(obs, packet)
            if not any(evidence_index[e][1].get('receptor_serial')==params['receptor_serial']
                       and evidence_index[e][1].get('residue_id')==params['residue_id'] for e in obs['evidence_ids']):
                raise ValueError('Receptor pair must be localized by cited clash evidence')
        if kind == 'mmff_strain' and atoms != valid:
            raise ValueError('MMFF strain requires the complete ordered molecular graph')
        if kind == 'mmff_strain':
            ready=packet['steering']['chemical_readiness'][view]
            if any(ready.get(k,{}).get('value') is not True for k in
                   ('graph_valid','protonation_validated','mmff_applicability_validated')):
                raise ValueError('MMFF needs validated graph, protonation and force-field applicability')
    return {x['observable_id']: OBSERVABLES[x['kind']][1] for x in observables}


def receptor_reference(obs, packet):
    params = obs['parameters']
    matches = [a for a in packet['steering']['receptor_atoms']
               if a['serial'] == params['receptor_serial'] and a['residue_id'] == params['residue_id']]
    if len(matches) != 1:
        raise ValueError('Receptor reference does not identify exactly one bound atom')
    return matches[0]['coords']


def validate_expression(tree, units):
    count, seen = 0, set()

    def visit(node, depth=0):
        nonlocal count
        count += 1
        if count > 128 or depth > 12 or not isinstance(node, dict):
            raise ValueError('Expression is too large, deep or malformed')
        op = node.get('op')
        if op == 'observable':
            if set(node) != {'op', 'id'} or node['id'] not in units:
                raise ValueError('Unknown observable reference')
            seen.add(node['id'])
            return UNITS[units[node['id']]]
        if op == 'constant':
            if (set(node) != {'op', 'value', 'unit', 'origin'} or not finite_number(node['value'])
                    or node['unit'] not in UNITS or not isinstance(node['origin'], str)
                    or len(node['origin'].strip()) < 5):
                raise ValueError('Constants need a finite value, unit and provenance')
            return UNITS[node['unit']]
        unary = {'relu', 'abs', 'sqrt', 'sin', 'cos', 'power'}
        binary = {'add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum', 'periodic_difference'}
        reduce = {'sum', 'mean'}
        if op not in unary | binary | reduce:
            raise ValueError('Unsupported expression operator')
        required = {'op', 'args', 'exponent'} if op == 'power' else {'op', 'args'}
        args = node.get('args')
        if set(node) != required or not isinstance(args, list) or not 1 <= len(args) <= 32:
            raise ValueError('Invalid operator fields or arguments')
        if op in unary and len(args) != 1 or op in binary and len(args) != 2:
            raise ValueError('Invalid operator arity')
        dims = [visit(a, depth+1) for a in args]
        if op in {'add', 'subtract', 'maximum', 'minimum', 'sum', 'mean'}:
            if any(d != dims[0] for d in dims):
                raise ValueError('Cannot combine different physical dimensions')
            return dims[0]
        if op == 'multiply':
            return tuple(a+b for a, b in zip(*dims))
        if op == 'divide':
            return tuple(a-b for a, b in zip(*dims))
        if op in {'sin', 'cos', 'periodic_difference'}:
            if any(d != UNITS['radian'] for d in dims):
                raise ValueError('Periodic operations require radians')
            return UNITS['radian'] if op == 'periodic_difference' else UNITS['dimensionless']
        if op in {'power', 'sqrt'}:
            p = node.get('exponent', .5)
            if not finite_number(p) or not .5 <= p <= 8:
                raise ValueError('Exponent must be finite in [0.5, 8]')
            return tuple(a*p for a in dims[0])
        return dims[0]

    if visit(tree) != UNITS['dimensionless'] or seen != set(units):
        raise ValueError('Each local objective must be dimensionless and use all declared observables')


def evaluate_expression(tree, values):
    template = next(iter(values.values()))

    def evaluate(node):
        op = node['op']
        if op == 'observable':
            return values[node['id']]
        if op == 'constant':
            # A degree constant converts to radians at this boundary.
            value = node['value'] * (math.pi/180 if node['unit'] == 'degree' else 1)
            return template.new_tensor(value)
        args = [evaluate(x) for x in node['args']]
        x = args[0]
        if op == 'add': result = x+args[1]
        elif op == 'subtract': result = x-args[1]
        elif op == 'multiply': result = x*args[1]
        elif op == 'divide':
            if abs(float(args[1].detach())) < 1e-12:
                raise ValueError('Expression denominator is zero')
            result = x/args[1]
        elif op == 'relu': result = torch.relu(x)
        elif op == 'abs': result = x.abs()
        elif op in {'sqrt', 'power'}:
            p = node.get('exponent', .5)
            if (float(x.detach()) < 0 and p != int(p)) or (float(x.detach()) == 0 and p < 1):
                raise ValueError('Power value or derivative is undefined')
            result = x.pow(p)
        elif op == 'sin': result = x.sin()
        elif op == 'cos': result = x.cos()
        elif op == 'periodic_difference': result = torch.atan2((x-args[1]).sin(), (x-args[1]).cos())
        elif op == 'maximum': result = torch.maximum(x, args[1])
        elif op == 'minimum': result = torch.minimum(x, args[1])
        elif op == 'sum': result = torch.stack(args).sum()
        elif op == 'mean': result = torch.stack(args).mean()
        else: raise ValueError('Unsupported expression operator')
        if result.ndim or not torch.isfinite(result):
            raise ValueError('Nonfinite or nonscalar expression')
        return result
    value = evaluate(tree)
    if not torch.isfinite(value) or float(value.detach()) < -1e-10:
        raise ValueError('Local objectives must be finite nonnegative deficits')
    return value


def observable_value(obs, coords, atom_ids, packet, mmff=None, molecule=None):
    index = {a: i for i, a in enumerate(atom_ids)}
    q = [coords[index[a]] for a in obs['atom_ids']]
    kind, params = obs['kind'], obs['parameters']

    def unit(v):
        length = v.norm()
        if float(length.detach()) < 1e-10:
            raise ValueError('Degenerate geometry')
        return v/length

    if kind in ('distance', 'receptor_distance', 'anchor_offset'):
        other = q[1] if kind == 'distance' else coords.new_tensor(
            receptor_reference(obs, packet) if kind == 'receptor_distance' else params['reference'])
        delta = q[0]-other
        if kind != 'anchor_offset' and float(delta.norm().detach()) < 1e-10:
            raise ValueError('Coincident distance has no radial derivative')
        return torch.linalg.vector_norm(delta)
    if kind == 'angle':
        a, b = unit(q[0]-q[1]), unit(q[2]-q[1])
        cosine = a.dot(b)
        if abs(float(cosine.detach())) >= 1-1e-10:
            raise ValueError('Collinear angle is not differentiable')
        return cosine.acos()
    if kind == 'direction_alignment':
        return unit(q[1]-q[0]).dot(unit(coords.new_tensor(params['reference'])))
    if kind == 'dihedral':
        axis = unit(q[2]-q[1])
        a, b = q[0]-q[1], q[3]-q[2]
        v, w = unit(a-a.dot(axis)*axis), unit(b-b.dot(axis)*axis)
        return torch.atan2(torch.linalg.cross(axis, v).dot(w), v.dot(w))
    if kind == 'signed_volume':
        return (q[1]-q[0]).dot(torch.linalg.cross(q[2]-q[0], q[3]-q[0]))
    if kind == 'mmff_strain':
        if mmff is None or molecule is None:
            raise ValueError('MMFF requires a validated stable graph and hydrogen preparation')
        return mmff.tensor(coords, molecule)
    raise ValueError('Unknown observable')


def aggregate_objectives(values, strategy):
    if strategy['mode'] == 'common_descent':
        # Reporting only. The controller differentiates each objective separately.
        return torch.stack(list(values.values())).max()
    aggregation = strategy['aggregation']
    items = torch.stack(list(values.values()))
    if aggregation['op'] == 'single': return items[0]
    if aggregation['op'] == 'maximum': return items.max()
    if aggregation['op'] == 'lp_norm':
        p = aggregation['p']
        return (items.pow(p).mean()+1e-12).pow(1/p)-1e-12**(1/p)
    raise ValueError('Unsupported cross-direction strategy')
