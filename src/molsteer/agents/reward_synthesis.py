"""Constructive reward workspace: source mechanisms -> local shapes -> trials.

These helpers produce drafts and measurements, never approve execution. The
expert chooses targets, scales, source adaptations and composition. The host
handles repetitive AST/lineage serialization and derivative calculation.
"""
from copy import deepcopy
import math
import torch
from molsteer.molthinker.expressions import (OBSERVABLES, validate_observables,
    validate_expression, evaluate_expression)
from .audit_contracts import ReferenceParameter


# Transfer advice, not alternative equations attributed to the source table.
TRANSFER = {
    'G01': ('set_distance', 'Specify the acceptable interval or one-sided set first; its normalized distance can be linear or quadratic. A tolerance is an input, not inferred from the alert.'),
    'G02': ('exclusion', 'Keep only penetration pressure; no attraction after separation. Derive the bound from identified radii/buffer, not the current closest distance.'),
    'G03': ('preservation', 'Preserve a justified anchor in the verified frame. A quadratic is a center target, not an interval or a whole-pose stability energy.'),
    'G04': ('orientation', 'Separate partner/feature typing from orientation. Unit-vector agreement requires a reference direction; a contact distance cannot replace it.'),
    'G05': ('signed_periodic_geometry', 'Use a signed margin for handedness and circular difference for torsion. A volume hinge is a new surrogate of the reported sign predicate, not the reported angle energy.'),
    'G06': ('assignment_shape', 'Transfer the reference point set, alignment and matching operator together. Nearest-distance attraction is not an occupancy/transport objective.'),
    'G07': ('environment_descriptor', 'Retain descriptor and reference library plus the forward/reverse coverage distinction. Raw pair distances are not this descriptor kernel.'),
    'P01': ('coupled_energy', 'Retain the complete typed molecular energy for coupled strain, or explicitly derive a local surrogate. Do not infer total energy from one stretched bond.'),
    'P02': ('typed_interface_energy', 'Transfer typed repulsion and attraction together; keep short-range behavior and frozen receptor. A distance interval is a different pilot surrogate.'),
    'P03': ('electrostatics', 'Charge products determine attraction/repulsion; transfer charge, dielectric and typing inputs. A shorter distance is not always beneficial.'),
    'P04': ('stationarity', 'Residual force is distinct from energy. Select a ready differentiable or zero-order path; a force norm needs oracle evaluations, not guessed energy values.'),
    'P05': ('binding_score', 'Retain surface distances, atom typing and score terms. Use as a terminal oracle when the suffix derivative is unavailable.'),
    'P06': ('area_target', 'A prepared group-area target and differentiable SASA can define a deficit. Missing area derivatives remain an evaluator-extension opportunity.'),
    'P07': ('field_similarity', 'Transfer charges, aligned surface points and reference field. A few receptor contacts cannot certify electrostatic surface similarity.'),
    'S01': ('population_target', 'Gaussian target weighting acts on trajectories; its local coordinate use needs a valid derivative and an explicitly new role.'),
    'S02': ('population_enrichment', 'Particle softmax allocates replication probability. It does not justify a within-molecule deficit aggregator.'),
    'S03': ('selectivity', 'Compare matched on/off-target evaluations and their direction. Missing off-target data cannot be replaced by a pocket-distance term.'),
    'S04': ('oracle_gradient_estimator', 'Use paired perturbations of a re-evaluable oracle within a budget; uncertainty and estimator variance are part of the design.'),
    'S05': ('population_pareto', 'Non-dominance retains tradeoffs in a population. Single-molecule gradient comparison is a separate constructed operator.'),
    'S06': ('depth_selection', 'Needs calibrated depth batches; do not interpret a batch integrity statistic as an individual coordinate gradient.'),
    'S07': ('discrete_feasibility', 'Preserve checker and population roles. Discrete validity predicates need proposal/search paths, not coordinate derivatives.')}

SHAPES = {
    'current_reference_quadratic': (['scale'], ['G01', 'G02', 'P01']),
    'interval_linear': (['lower', 'upper', 'scale'], ['G01']),
    'interval_quadratic': (['lower', 'upper', 'scale'], ['G01']),
    'lower_linear': (['lower', 'scale'], ['G01', 'G02']),
    'lower_quadratic': (['lower', 'scale'], ['G01', 'G02']),
    'upper_linear': (['upper', 'scale'], ['G01', 'G05', 'P01']),
    'upper_quadratic': (['upper', 'scale'], ['G01', 'G05', 'P01']),
    'center_quadratic': (['target', 'scale'], ['G01', 'G03']),
    'periodic_quadratic': (['target', 'scale'], ['G01', 'G05']),
    'periodic_cosine': (['target', 'scale'], ['G01', 'G05']),
    'alignment_linear': (['scale'], ['G04']),
    'signed_margin': (['sign', 'margin', 'scale'], ['G05'])}

SYNTHESIS_GUIDE = {
    'workflow': ['Read ranked biological goals, their full configured raw trajectories, preservation and mechanism dependencies.',
        'Inspect source mechanisms, identify what transfers and what must change.',
        'Determine target sets or justified optimization relations and explicit unknown parameters.',
        'Derive local response (direction, active/stopping regions, curvature and coupling) before selecting construction operators.',
        'Construct relation-level potentials; consolidate duplicate relations within each mechanism.',
        'Compare compositions by acceptable-set semantics, marginal response and measured copy-coordinate interactions.',
        'Use existing final numerical validation and exact submission once the candidate is ready.'],
    'shape_options': 'Source cards expose relevant serialization options after retrieval; custom drafts remain available. The helper shape menu does not choose scientific targets.',
    'parameter_record': {'value': 'finite number', 'unit': 'optional; defaults to observable unit, sign is dimensionless',
        'origin': 'unique explanation', 'role': 'optional; inferred from parameter name: target/reference, bounds/tolerance, scale/normalization, sign/operator_parameter',
        'provenance': 'observed_reference/source_formula/calibration/declared_assumption/diagnostic_screening',
        'evidence_ids': 'actual measured IDs or []', 'source_locators': 'actual inspected locations or []',
        'derivation': 'how this input was obtained; do not infer calibration from a screening cutoff'},
    'relation_format': {'observable': 'bound observable object', 'function_id': 'actual inspected source function',
        'shape': 'one shape name', 'parameters': 'map from parameter names to records above',
        'clause_ids': 'biological clauses this relation actually implements'},
    'within_direction': 'single / maximum / sum. sum is a derived sum of nonnegative relation deficits with intersection zero set, not fixed competing objective weights. Explain response allocation.',
    'allocation_tool': 'priority_weights carries value ranking into the actual scalar weighted_sum/maximum/lp_norm. Preservation belongs in the scalar or independent evaluation. Default rank_decay is a configurable ordinal preference, not measured efficacy. derive_allocation_operator can additionally derive a norm exponent from a desired marginal response.',
    'dynamic_relations': 'Use current_reference_quadratic with bond_length_error, bond_angle_error or typed_steric_overlap. The evaluator resolves current chemistry; only normalization is supplied as a constant. Whole-graph identity is not an activation condition.',
    'construction_scope': 'Host serializes the expert-selected mathematical construction; generated probes describe that construction, not biological calibration.',
    'scope': 'Draft assistance; no automatic objective selection, missing-data substitution or new submission gate.'}


def derive_allocation_response(first_deficit, second_deficit, desired_pressure_ratio, reason):
    """Invert an expert-chosen marginal law instead of defaulting to a norm p.

    For the mean lp norm, dF/dF1 divided by dF/dF2 is (F1/F2)^(p-1).
    The epsilon used by the executor cancels in this ratio. This is a new
    composition derivation, not a formula attributed to the function table.
    """
    values = (first_deficit, second_deficit, desired_pressure_ratio)
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 < v <= 1e12 for v in values):
        return {'status': 'needs_input', 'blocking': False,
            'hint': 'Supply two positive dimensionless deficits and a positive desired marginal-pressure ratio. Zero/unknown deficits do not identify a norm exponent.'}
    if not isinstance(reason, str) or not reason.strip():
        return {'status': 'needs_input', 'blocking': False,
            'hint': 'Explain why this local response ratio is desired; biological handoff order alone is not a reward coefficient.'}
    ratio = first_deficit/second_deficit
    if math.isclose(ratio, 1., rel_tol=1e-8):
        return {'status': 'underdetermined', 'blocking': False,
            'hint': 'Equal deficits have equal marginal pressure for every unweighted lp norm. This observation cannot identify p or express unequal pressure.'}
    p = 1+math.log(desired_pressure_ratio)/math.log(ratio)
    candidate = None
    if 1 < p <= 8:
        candidate = dict(mode='scalar_potential', aggregation={'op': 'lp_norm', 'p': p},
            justification=f'New response-law composition: at deficits {first_deficit}/{second_deficit}, require sensitivity ratio {desired_pressure_ratio}; solving p=1+log(q)/log(F1/F2) gives {p}. {reason}')
    return {'status': 'derived_candidate' if candidate else 'outside_current_execution_family',
        'blocking': False, 'derived_p': p, 'candidate_strategy': candidate,
        'derivation': 'For A=(mean(F_i^p)+epsilon)^(1/p)-epsilon^(1/p), alpha_i=F_i^(p-1)*(mean(F_j^p)+epsilon)^(1/p-1)/n. Thus alpha_1/alpha_2=(F1/F2)^(p-1).',
        'zero_set': 'All retained nonnegative deficits equal zero; one improvement cannot make another failed predicate zero.',
        'scope': 'A local marginal response design point, not a fixed ratio across all states or a proof of live feasibility/terminal utility.',
        'warnings': ['The function table grounds local mechanisms; this composition is newly derived.',
            'Normalization changes deficit ratios. Verify scales and measured projected gradients before choosing.',
            'Large p can attenuate a small unresolved deficit; inspect the range, not only this design point.',
            'An unsupported p is not silently clamped or replaced by a familiar template.']}


def function_card(record):
    ident = record.get('function_id')
    mechanism, transfer = TRANSFER.get(ident, ('unreviewed', 'Inspect the original relation before adapting.'))
    return {k: deepcopy(record.get(k)) for k in ('function_id', 'name_en', 'chunk_id', 'source_id',
        'formula', 'role', 'prerequisites')} | {'mechanism': mechanism, 'transfer_reasoning': transfer,
        'constructible_shapes': [s for s, (_, ids) in SHAPES.items() if ident in ids],
        'serialization_options': {s: {'parameters': fields} for s, (fields, ids) in SHAPES.items() if ident in ids}}


def _op(name, *args, **extra):
    return {'op': name, 'args': list(args), **extra}


def _build_relation(relation, source):
    obs, shape = relation['observable'], relation['shape']
    needed, sources = SHAPES[shape]
    if source['function_id'] not in sources:
        raise ValueError('shape_source')
    kind = obs['kind']
    if shape.startswith('periodic') and kind != 'dihedral':
        raise ValueError('periodic_kind')
    if shape == 'signed_margin' and kind != 'signed_volume':
        raise ValueError('signed_kind')
    if shape == 'alignment_linear' and kind != 'direction_alignment':
        raise ValueError('alignment_kind')
    unit = OBSERVABLES[kind][1]
    params = {}
    for key in needed:
        record = deepcopy(relation['parameters'][key])
        record.setdefault('unit', 'dimensionless' if key == 'sign' else unit)
        record.setdefault('role', 'normalization' if key == 'scale' else 'reference' if key == 'target' else 'operator_parameter' if key == 'sign' else 'tolerance')
        record.setdefault('evidence_ids', [])
        record.setdefault('source_locators', [])
        params[key] = ReferenceParameter.model_validate(record).model_dump()
    if any(p['unit'] != ('dimensionless' if k == 'sign' else unit) for k, p in params.items()):
        raise ValueError('parameter_unit')
    if any(not math.isfinite(p['value']) for p in params.values()) or params['scale']['value'] <= 0:
        raise ValueError('scale')
    if shape.startswith('interval') and params['lower']['value'] >= params['upper']['value']:
        raise ValueError('interval')
    if shape.startswith('periodic') and not 0 < params['scale']['value'] < math.pi:
        raise ValueError('circular_scale')
    if shape == 'signed_margin' and (params['sign']['value'] not in (-1, 1) or params['margin']['value'] <= 0):
        raise ValueError('signed_margin')
    coefficients = []
    def coefficient(value):
        origin = f'Exact algebraic coefficient {value} in {obs["observable_id"]} {shape} specialization'
        conventional_half = value == .5 and source['function_id'] not in ('G01', 'G02', 'G03')
        coefficients.append(dict(origin=origin, value=value, unit='dimensionless', role='operator_parameter',
            provenance='declared_assumption' if conventional_half else 'source_formula', evidence_ids=[],
            source_locators=[] if conventional_half else [source['chunk_id']],
            derivation='Half-squared curvature convention chosen by the construction, not a reported coefficient or physical calibration.' if conventional_half else
                'Exact algebraic coefficient in the constructed normalization of the inspected relation; not a fitted physical parameter.'))
        return {'op': 'constant', 'value': value, 'unit': 'dimensionless', 'origin': origin}
    def c(key):
        p = params[key]
        return {'op': 'constant', **{k: p[k] for k in ('value', 'unit', 'origin')}}
    z = {'op': 'observable', 'id': obs['observable_id']}
    scale = c('scale')
    name = obs['observable_id']
    number = lambda key: str(params[key]['value'])
    if shape == 'current_reference_quadratic':
        if kind not in ('bond_length_error', 'bond_angle_error', 'typed_steric_overlap'):
            raise ValueError('current_reference_kind')
        deficit = _op('divide', z, scale)
        zero, base = f'{name} = 0 under its current chemical reference', 0.
        samples = ([params['scale']['value'], 2*params['scale']['value']] if kind == 'typed_steric_overlap'
                   else [-params['scale']['value'], params['scale']['value']])
        formula = f'{name}(X,current_chemistry)/{number("scale")}'
    elif shape.startswith('interval'):
        deficit = _op('divide', _op('add', _op('relu', _op('subtract', c('lower'), z)),
                      _op('relu', _op('subtract', z, c('upper')))), scale)
        zero = f'{params["lower"]["value"]} <= {obs["observable_id"]} <= {params["upper"]["value"]} {unit}'
        base = (params['lower']['value']+params['upper']['value'])/2
        samples = [params['lower']['value']-params['scale']['value'], params['upper']['value']+params['scale']['value']]
        formula = f'([{number("lower")}-{name}]_+ + [{name}-{number("upper")}]_+)/{number("scale")}'
    elif shape.startswith('lower') or shape.startswith('upper'):
        lower = shape.startswith('lower')
        bound = params['lower' if lower else 'upper']['value']
        deficit = _op('divide', _op('relu', _op('subtract', c('lower'), z) if lower else _op('subtract', z, c('upper'))), scale)
        zero = f'{obs["observable_id"]} {">=" if lower else "<="} {bound} {unit}'
        base = bound
        sign = -1 if lower else 1
        samples = [bound+sign*params['scale']['value'], bound+2*sign*params['scale']['value']]
        formula = (f'[{bound}-{name}]_+' if lower else f'[{name}-{bound}]_+') + f'/{number("scale")}'
    elif shape == 'center_quadratic' or shape.startswith('periodic'):
        residual = _op('periodic_difference' if shape.startswith('periodic') else 'subtract', z, c('target'))
        deficit = _op('divide', residual, scale)
        zero = f'{obs["observable_id"]} = {params["target"]["value"]} {unit}' + (' modulo 2*pi' if shape.startswith('periodic') else '')
        base = params['target']['value']
        samples = [base-params['scale']['value'], base+params['scale']['value']]
        difference = f'wrap({name}-{number("target")})' if shape.startswith('periodic') else f'({name}-{number("target")})'
        formula = f'{difference}/{number("scale")}'
        if shape == 'periodic_cosine':
            deficit = _op('divide', _op('subtract', coefficient(1), _op('cos', residual)),
                          _op('subtract', coefficient(1), _op('cos', scale)))
            formula = f'(1-cos({difference}))/(1-cos({number("scale")}))'
    elif shape == 'alignment_linear':
        deficit = _op('divide', _op('subtract', coefficient(1), z), scale)
        zero, base, samples = f'{obs["observable_id"]} = 1 (aligned unit vectors)', 1., [0., -1.]
        formula = f'(1-{name})/{number("scale")}'
    else:
        deficit = _op('divide', _op('relu', _op('subtract', c('margin'), _op('multiply', c('sign'), z))), scale)
        margin, sign = params['margin']['value'], params['sign']['value']
        zero = f'{sign}*{obs["observable_id"]} >= {margin} {unit}'
        base, samples = sign*margin, [0., -sign*margin]
        formula = f'[{margin}-{sign}*{name}]_+/{number("scale")}'
    if shape.endswith('quadratic'):
        deficit = _op('multiply', coefficient(.5), _op('power', deficit, exponent=2))
        formula = f'0.5*({formula})^2'
    if kind in ('distance', 'receptor_distance', 'anchor_offset', 'angle'):
        lo, hi = 0., math.pi if kind == 'angle' else math.inf
        if not lo <= base <= hi:
            raise ValueError('physical_domain')
        samples = [v for v in samples if lo <= v <= hi]
        # Near a physical endpoint choose attainable probes on the other side;
        # never invent negative distances to demonstrate a useful derivative.
        for v in (base-params['scale']['value'], base+params['scale']['value'],
                  base-2*params['scale']['value'], base+2*params['scale']['value']):
            if len(samples) >= 2:
                break
            inside_target = (params['lower']['value'] <= v <= params['upper']['value'] if shape.startswith('interval') else
                v >= params['lower']['value'] if shape.startswith('lower') else
                v <= params['upper']['value'] if shape.startswith('upper') else v == base)
            if lo <= v <= hi and v != base and v not in samples and not inside_target:
                samples.append(v)
        if len(samples) < 2:
            raise ValueError('physical_domain')
    if shape.startswith('periodic'):
        base = math.atan2(math.sin(base), math.cos(base))
        samples = [math.atan2(math.sin(v), math.cos(v)) for v in samples]
    new = shape in ('periodic_cosine', 'signed_margin') or source['function_id'] == 'P01' or (
        source['function_id'] == 'G05' and shape != 'upper_linear')
    return dict(expression=deficit, formula=formula, parameters=list(params.values())+coefficients, zero_set=zero,
        baseline=base, samples=samples, source_transform='new_surrogate' if new else 'specialized',
        derivation=f'{shape}: construct the normalized relation deficit with zero set {zero}. ' +
            ('New mathematical surrogate of the inspected mechanism; not the reported energy.' if new else 'Specialization of the inspected local shape.'))


def construct_direction(biology_direction, relations, within_direction, retrieved, packet, interpretation=None):
    """Build a field-ready draft, returning constructive feedback on partial inputs."""
    hints = {'shape_source': 'Use an inspected candidate source for this shape, or propose your custom expression with the existing draft tools.',
        'periodic_kind': 'Circular shapes require a dihedral observable in radians.', 'signed_kind': 'A signed margin requires a signed_volume observable.',
        'alignment_kind': 'Alignment needs a direction_alignment observable with a measured reference direction.',
        'parameter_unit': 'Each target/scale uses the observable unit; sign uses dimensionless.',
        'scale': 'Supply finite targets and a positive normalization scale with its origin.',
        'interval': 'Supply lower < upper; do not derive them from the current abnormal value.',
        'circular_scale': 'Supply a positive scale below pi radians for this local circular construction.',
        'signed_margin': 'Supply sign +1/-1 and a positive geometric preservation margin.',
        'physical_domain': 'Choose attainable target/probe values in the observable domain; negative distances and vacuous one-sided sets cannot demonstrate a meaningful repair.',
        'duplicate_observable': 'Consolidate repeated observables into one relation or use a custom draft to model coupled terms.',
        'operator': 'Choose single for one relation, maximum for worst violation, or sum for intersection deficits and explain allocation.'}
    hints['current_reference_kind'] = 'Use bond_length_error, bond_angle_error or typed_steric_overlap for current_reference_quadratic.'
    try:
        if not isinstance(relations, list) or not 1 <= len(relations) <= 5:
            return {'status': 'needs_input', 'blocking': False, 'hint': 'Supply 1-5 relations per direction; larger/custom constructions can use the existing expression draft tools.'}
        obs = [r['observable'] for r in relations]
        if any(not isinstance(o, dict) for o in obs):
            return {'status':'needs_input','blocking':False,
                    'hint':'relation.observable must be a JSON object, not an ID or prose. Use the bound observable schema from get_expert_contract.',
                    'observable_fields':['observable_id','kind','view','atom_ids','evidence_ids','parameters']}
        observable_fields = {'observable_id', 'kind', 'view', 'atom_ids', 'evidence_ids', 'parameters'}
        for index, (relation, observable) in enumerate(zip(relations, obs)):
            if set(observable) != observable_fields:
                return {'status':'needs_input', 'blocking':False, 'relation_index':index,
                        'validation_path':['relations', index, 'observable'],
                        'missing_fields':sorted(observable_fields-set(observable)),
                        'unexpected_fields':sorted(set(observable)-observable_fields),
                        'observable_fields':sorted(observable_fields),
                        'hint':'Supply exactly the bound observable fields. kind is an OBSERVABLES key; observable_id is your unique ID. Put units only in parameter records; use parameters={} for bond/angle errors and cite current direction evidence.'}
            shape = relation.get('shape')
            if shape not in SHAPES:
                return {'status':'needs_input', 'blocking':False,
                        'validation_path':['relations', index, 'shape'],
                        'hint':'Select an inspected function card serialization option.',
                        'shape_options':list(SHAPES)}
            parameters = relation.get('parameters')
            required = SHAPES[shape][0]
            if not isinstance(parameters, dict) or set(parameters) != set(required):
                return {'status':'needs_input', 'blocking':False,
                        'validation_path':['relations', index, 'parameters'],
                        'required_parameters':required,
                        'missing_fields':sorted(set(required)-set(parameters or {})),
                        'unexpected_fields':sorted(set(parameters or {})-set(required)),
                        'parameter_record':deepcopy(SYNTHESIS_GUIDE['parameter_record']),
                        'hint':'Use these exact parameter names and units. current_reference_quadratic only takes scale in the observable unit; chemistry supplies its current reference. A force constant has different units and cannot serve directly as a length scale.'}
        if len({o['observable_id'] for o in obs}) != len(obs):
            raise ValueError('duplicate_observable')
        units = validate_observables(obs, packet, set(biology_direction['evidence_ids']))
        sources = {row['function_id']: (ret, row) for ret in retrieved for row in ret['records'] if row.get('function_id')}
        parts = [_build_relation(r, sources[r['function_id']][1]) for r in relations]
        if within_direction not in ('single', 'maximum', 'sum') or (within_direction == 'single' and len(parts) != 1):
            raise ValueError('operator')
        expression = parts[0]['expression']
        if within_direction == 'sum':
            expression = _op('sum', *(p['expression'] for p in parts))
        elif within_direction == 'maximum':
            # Balanced serialization keeps the existing depth limit intact.
            layer = [p['expression'] for p in parts]
            while len(layer) > 1:
                layer = [_op('maximum', layer[i], layer[i+1]) if i+1 < len(layer) else layer[i]
                         for i in range(0, len(layer), 2)]
            expression = layer[0]
        validate_expression(expression, units)
        parameters = {}
        for part in parts:
            for param in part['parameters']:
                key = (param['origin'], param['unit'], param['value'])
                if key in parameters and parameters[key] != param:
                    previous=parameters[key]
                    if any(previous[field] != param[field] for field in ('role','provenance')):
                        return {'status': 'needs_input', 'blocking': False,
                            'parameter_origin':param['origin'],
                            'conflicting_fields':[field for field in ('role','provenance') if previous[field]!=param[field]],
                            'hint': 'The same constant has conflicting provenance or roles. Reconcile these records; do not infer calibration.'}
                    # The same source constant can support several relations.
                    # Retain all corroboration rather than rejecting distinct
                    # explanations of an otherwise identical physical input.
                    for field in ('evidence_ids','source_locators'):
                        previous[field]=list(dict.fromkeys(previous[field]+param[field]))
                    if param['derivation'] != previous['derivation']:
                        previous['derivation']+='; '+param['derivation']
                    continue
                parameters[key] = param
        baseline = {o['observable_id']: p['baseline'] for o, p in zip(obs, parts)}
        probe_values = [baseline]
        for o, part in zip(obs, parts):
            probe_values += [dict(baseline, **{o['observable_id']: v}) for v in part['samples']]
        probes = []
        for values in probe_values:
            tensors = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in values.items()}
            loss = evaluate_expression(expression, tensors)
            gradients = torch.autograd.grad(loss, tuple(tensors.values()), allow_unused=True)
            probes.append(dict(observable_values=values, expected_zero=values == baseline,
                derivative_signs={k: 'zero' if g is None or abs(float(g)) <= 1e-10 else 'positive' if float(g) > 0 else 'negative' for k, g in zip(tensors, gradients)},
                reason='Constructed target-set example; derivative signs measured by automatic differentiation, not a biological calibration test.'))
        selected = list(dict.fromkeys(r['function_id'] for r in relations))
        lineage = [dict(source_id=sources[f][1]['source_id'], locator=sources[f][1]['chunk_id'], original_formula=sources[f][1]['formula'],
            adaptation='; '.join(p['derivation'] for r, p in zip(relations, parts) if r['function_id'] == f)) for f in selected]
        coverage = {}
        for relation in relations:
            for clause in relation.get('clause_ids', []):
                coverage.setdefault(clause, []).append(relation['observable']['observable_id'])
        draft = dict(direction_id=biology_direction['direction_id'], status='executable',
            retrieval_ids=list(dict.fromkeys(sources[f][0]['retrieval_id'] for f in selected)),
            source_ids=list(dict.fromkeys(x['source_id'] for x in lineage)), function_lineage=lineage,
            formula=('F = ' + parts[0]['formula'] if within_direction == 'single' else
                     'F = ' + ('max' if within_direction == 'maximum' else 'sum') + '(' + ', '.join(p['formula'] for p in parts) + ')'),
            derivation_summary='Construct each relation from its target set, then use the expert-selected within-mechanism operator. ' + '; '.join(p['derivation'] for p in parts),
            acceptable_set='Intersection of the following constructed local sets: ' + '; '.join(p['zero_set'] for p in parts),
            normalization='Each scale is supplied by the expert with explicit parameter provenance; no scale is inferred from alert severity.',
            assumptions=['Targets and mechanism choices require expert review; construction does not establish calibration or terminal benefit.'],
            alternatives=['Compare a different local curvature or a complete ready physical evaluator before selecting this draft.'],
            gradient_path='Coordinate-copy derivative; generator pullback and native continuation are not_run.',
            marginal_sensitivity='Linear/quadratic/circular local response as constructed. ' + ('All positive deficits receive additive pressure; correlated measurements must be consolidated.' if within_direction == 'sum' else 'Worst violation receives pressure; test switching and starvation.' if within_direction == 'maximum' else 'No cross-relation allocation.'),
            failure_mode='Local deficits can decrease without improving independent terminal validity or binding; native dynamics or changed chemistry may invalidate them.',
            missing_requirements=[], observables=obs, expression=expression,
            source_transform='new_surrogate' if any(p['source_transform'] == 'new_surrogate' for p in parts) else 'specialized',
            function_basis=[dict(locator=x['locator'], role='preservation' if biology_direction['disposition'] == 'constraint' else 'direct_repair',
                decision='selected', reason=x['adaptation'], missing_requirements=[]) for x in lineage],
            reference_parameters=list(parameters.values()), shape_probes=probes,
            predicate_coverage=[dict(clause_id=k, observable_ids=v, implementation='The constructed expression contains these measured relation deficits; review their target sets against the biological clause.') for k, v in coverage.items()])
        narrative = {'derivation_summary', 'assumptions', 'alternatives', 'normalization', 'marginal_sensitivity', 'failure_mode', 'constraint_mode'}
        if interpretation:
            draft.update({k: deepcopy(v) for k, v in interpretation.items() if k in narrative})
        return {'status': 'draft_only', 'blocking': False, 'validated': False, 'direction': draft,
            'authorship': 'Expert chose relations, targets, scales and operator; host constructed AST, exact source copies and derivative examples.',
            'next_step': 'Review scientific targets and compare measured candidate interactions; final existing full-design test still owns execution approval.'}
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {'status': 'needs_input', 'blocking': False, 'error_type': type(exc).__name__,
            'validation_error':str(exc),
            'hint': hints.get(str(exc), 'Inspect the shape guide, complete parameter records and bound observable support. Unsupported inputs remain explicit; use custom draft tools if the intended mechanism is not representable.')}


def preview_architectures(packet, biology, drafts, dynamics, strategies):
    """Measure interactions on copies; mixed views are never a live conflict test."""
    from molsteer.molexecutor.expert_control import ExpertEvaluator, control_direction
    from molsteer.molthinker.expressions import aggregate_objectives
    if not drafts or not isinstance(strategies, list) or not 1 <= len(strategies) <= 6:
        return {'status': 'needs_input', 'blocking': False, 'hint': 'Construct drafts first and supply 1-6 strategy objects.'}
    try:
        directions = list(drafts.values())
        evaluator = ExpertEvaluator({'mathematical_design': {'directions': directions, 'strategy': {}}, 'biology_plan': biology}, packet)
        views = sorted({o['view'] for d in directions for o in d['observables']})
        if len(views) != 1:
            return {'status': 'advisory', 'blocking': False, 'scope': 'mixed_views_unmapped',
                'hint': 'Compare functions by target-set semantics; saved state/prediction coordinates lack a shared live pullback. Do not infer feasible common descent from independent coordinate blocks.'}
        view = views[0]; snapshot = packet['steering']['coordinate_snapshots'][view]
        variable = torch.tensor(snapshot['coords_angstrom'], dtype=torch.float64, requires_grad=True)
        values = evaluator.components({view: variable}); objectives = evaluator.objectives(values)
        if not objectives:
            return {'status': 'advisory', 'blocking': False, 'hint': 'Preservation alone does not define a soft repair objective.'}
        editable = dynamics.get('editable_atom_ids')
        support = {a for d in directions for o in d['observables'] for a in o['atom_ids']}
        mask = variable.new_tensor([editable is None or a in editable for a in snapshot['atom_ids']])[:, None].expand_as(variable)
        previews = []
        for strategy in strategies:
            try:
                if set(strategy) - {'mode', 'aggregation', 'justification', 'priority_weights', 'priority_basis'}:
                    raise ValueError('strategy')
                if strategy['mode'] != 'scalar_potential':
                    raise ValueError('strategy')
                agg = strategy['aggregation']
                if strategy['mode'] == 'scalar_potential' and (not isinstance(agg, dict) or agg.get('op') not in ('single', 'maximum', 'lp_norm', 'weighted_sum') or
                    (agg.get('op') == 'single' and len(objectives) != 1) or
                    (agg.get('op') == 'lp_norm' and (type(agg.get('p')) not in (int, float) or not 1 < agg['p'] <= 8))):
                    raise ValueError('strategy')
                direction, grads, audit = control_direction(objectives, variable, mask, strategy)
                sensitivities = None
                if strategy['mode'] == 'scalar_potential':
                    scalars = {k: v.detach().requires_grad_(True) for k, v in objectives.items()}
                    total = aggregate_objectives(scalars, strategy)
                    derivatives = torch.autograd.grad(total, tuple(scalars.values()), allow_unused=True)
                    sensitivities = {k: 0. if d is None else float(d) for k, d in zip(scalars, derivatives)}
                unit = direction/direction.norm().clamp(min=1e-30)
                constraint_effects = {}
                for ident, value in values.items():
                    if evaluator.roles[ident] == 'constraint':
                        g, = torch.autograd.grad(value, variable, retain_graph=True)
                        constraint_effects[ident] = float((g*mask*unit).sum())
                previews.append(dict(strategy=deepcopy(strategy), status=audit['status'],
                    deficits={k: float(v.detach()) for k, v in values.items()},
                    masked_gradient_norms={k: float(g.norm()) for k, g in zip(objectives, grads)},
                    copy_gradient_cosines=audit['cosine'], marginal_allocation=sensitivities,
                    objective_unit_step_derivatives={k: float((g*unit).sum()) for k, g in zip(objectives, grads)},
                    preservation_first_order_effects=constraint_effects,
                    limitation='First-order copy geometry only. Zero violation gradient does not certify finite-step preservation. No native continuation or terminal outcome was measured.'))
            except (ValueError, RuntimeError, KeyError, TypeError):
                previews.append({'status': 'needs_input', 'hint': 'Use supported strategy fields/aggregation and inspect stationary or undefined derivatives; this comparison never rejects a submission.'})
        return {'status': 'advisory', 'blocking': False, 'scope': 'coordinate_copy_same_view', 'view': view,
            'previews': previews, 'selected_strategy': None,
            'decision_rule': 'Expert selects by current-state repair mechanisms, declared goal coverage, local response, starvation/conflict and preservation. Copy loss does not measure future efficacy.'}
    except (ValueError, RuntimeError, KeyError, TypeError):
        return {'status': 'needs_input', 'blocking': False, 'hint': 'One or more observables/evaluators lack prerequisites. Compare available mechanisms and retain missing physics explicitly.', 'scope': 'coordinate_copy_only'}
