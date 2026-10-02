"""Evidence coverage, reference provenance and executable function-shape audits."""
import math
import re
from copy import deepcopy
from .audit_contracts import FACTORS
from .contract_errors import ContractValidationError
from molsteer.molreader.measurement_refs import measurement_references

FACTOR_METRICS = {
    'bond_geometry': {'bond_lengths', 'mmff_local_geometry'},
    'angle_torsion_stereochemistry': {'bond_angles', 'mmff_local_geometry', 'torsions',
                                    'ring_planarity', 'double_bond_planarity', 'stereochemistry'},
    'intramolecular_stability': {'mmff_energy', 'mmff_strain', 'intramolecular_clashes'},
    'steric_feasibility': {'protein_clashes', 'intramolecular_clashes', 'valence', 'connectivity'},
    'target_contacts': {'protein_contacts', 'hydrogen_bond_candidates', 'hydrophobic_contacts',
                        'salt_bridge_candidates', 'aromatic_contacts'},
    'target_surface_burial': {'burial_sasa', 'shape'},
    'chemical_identity_functional_groups': {'chemistry_context', 'atom_confidence', 'charge_confidence',
                                          'bond_confidence', 'structural_alerts'},
    'terminal_task_utility': {'affinity', 'qed', 'sa_score', 'logp', 'tpsa', 'vina_score'},
}
FACTOR_OBSERVABLES = {
    'bond_geometry': {'distance', 'mmff_strain'},
    'angle_torsion_stereochemistry': {'angle', 'dihedral', 'signed_volume', 'mmff_strain'},
    'intramolecular_stability': {'mmff_strain'},
    'steric_feasibility': {'distance', 'receptor_distance', 'mmff_strain'},
    'target_contacts': {'receptor_distance', 'direction_alignment'},
    # Area targets and categorical transitions have no implemented derivatives.
    'target_surface_burial': set(), 'chemical_identity_functional_groups': set(),
    'terminal_task_utility': set(),
}
CHEMICAL_REFERENCE_METRICS = {'mmff_local_geometry','bond_lengths','chemistry_context',
    'stereochemistry','ring_planarity','double_bond_planarity','protein_clashes','intramolecular_clashes'}


def bounded_values(value, limit=12):
    if isinstance(value, list):
        if len(value)<=limit:
            return [bounded_values(v,limit) for v in value]
        return {'items': [bounded_values(v, limit) for v in value[:limit]],
                'total_count': len(value), 'truncated': len(value) > limit}
    if isinstance(value, dict):
        return {k: bounded_values(v, limit) for k, v in value.items()}
    return value


def inspect_bound_measurements(measured, evidence_ids, include_details=False):
    if not evidence_ids or len(evidence_ids)>32 or len(set(evidence_ids))!=len(evidence_ids):
        raise ValueError('Use 1 to 32 distinct bound measurement IDs')
    for index, ident in enumerate(evidence_ids):
        if ident not in measured:
            raise ContractValidationError('Unknown bound measurement reference',['evidence_ids',index])
    records=[]
    for ident in evidence_ids:
        record=deepcopy(measured[ident])
        if record.get('kind')=='measured_summary' and not include_details:
            record['values']=bounded_values(record['values'])
            record['value_arrays_bounded']=True
        records.append(record)
    return {'measurements':records,'requested_count':len(records)}


def measurement_catalog(measured, metric_id, view='prediction', atom_id=None, offset=0, limit=12):
    """Page exact reference metadata; values stay available through ID inspection."""
    if view not in ('prediction', 'state') or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('Use state/prediction, a nonnegative offset and limit 1-32')
    records = [r for r in measured.values() if r.get('metric_id') == metric_id and r.get('view') == view
               and (atom_id is None or atom_id in r.get('atom_ids', []))]
    records.sort(key=lambda r: (r.get('kind') != 'derived_geometry', bool(r.get('path')), str(r.get('path'))))
    return {'references': [{k: deepcopy(v) for k, v in r.items() if k != 'values'} for r in records[offset:offset+limit]],
            'total_count': len(records), 'offset': offset,
            'next_offset': offset+limit if offset+limit < len(records) else None,
            'value_lookup': 'Use inspect_measurements with these exact evidence_id values. Summary arrays are bounded by default; include_details=true reads the complete stored summary.'}


def replace_draft_fields(draft, changes):
    import json
    if not changes or len(changes)>16:
        raise ValueError('Use 1 to 16 replacements of existing draft fields')
    result=deepcopy(draft)
    for index, change in enumerate(changes):
        path=change['path'];cursor=result
        for segment in path[:-1]:
            if not ((isinstance(cursor,dict) and type(segment) is str and segment in cursor) or
                    (isinstance(cursor,list) and type(segment) is int and 0<=segment<len(cursor))):
                raise ContractValidationError('Draft replacement path must identify an existing field',['changes',index,'path'])
            cursor=cursor[segment]
        last=path[-1]
        if not ((isinstance(cursor,dict) and type(last) is str and last in cursor) or
                (isinstance(cursor,list) and type(last) is int and 0<=last<len(cursor))):
            raise ContractValidationError('Draft replacement path must identify an existing field',['changes',index,'path'])
        cursor[last]=deepcopy(change['value'])
    json.dumps(result,allow_nan=False)
    return result


def stage_direction_draft(staged, direction, selected_ids):
    """Retain one model-authored draft; staging does not validate or execute it."""
    if not isinstance(direction, dict) or direction.get('direction_id') not in selected_ids:
        raise ContractValidationError('Stage a direction selected by the biological plan', ['direction', 'direction_id'])
    staged[direction['direction_id']] = deepcopy(direction)
    return {'status': 'draft_only', 'direction_id': direction['direction_id'],
            'staged_direction_ids': sorted(staged), 'validated': False,
            'next_step': 'Stage all selected directions, then test_staged_mathematical_design. Nothing is executable before the complete design passes every check.'}


def retain_proposed_directions(staged, design, selected_ids):
    """Keep model edits through reassembly; draft retention never approves them."""
    for direction in design.get('directions', []):
        if isinstance(direction,dict) and direction.get('direction_id') in selected_ids:
            staged[direction['direction_id']]=deepcopy(direction)


def assemble_direction_drafts(staged, biology, metadata):
    if set(metadata) != {'strategy', 'conflict_assessment', 'independent_evaluation', 'design_audit'}:
        raise ValueError('Staged design metadata must contain only the four global design fields')
    from .decision_workspace import selected_directions
    selected = [d['direction_id'] for d in selected_directions(biology)]
    if set(staged) != set(selected):
        raise ContractValidationError('Math must account for each selected direction without changing biological priorities', ['directions'])
    return {'kind': 'MathematicalDesign', 'schema_version': '2.0',
            'directions': [deepcopy(staged[ident]) for ident in selected], **deepcopy(metadata)}


def biophysical_context(packet, factor='all', view='both'):
    if factor != 'all' and factor not in FACTORS:
        raise ValueError('Unknown biophysical factor')
    if view not in ('both', 'state', 'prediction'):
        raise ValueError('Use state, prediction or both for current checkpoint factors')
    requested_views = {'state', 'prediction'} if view == 'both' else {view}
    factors = FACTORS if factor == 'all' else (factor,)
    references = measurement_references(packet)
    cards = []
    for name in factors:
        observations = [dict(metric_id=m['metric_id'], view=m['view'], status=m['status'],
            values=bounded_values(m['values'], 4), thresholds=bounded_values(m.get('thresholds', {}), 4),
            evidence=bounded_values(m.get('evidence', []), 4),
            measurement_references=measurement_catalog(references, m['metric_id'], m['view'])) for m in packet['observations']
            if m['metric_id'] in FACTOR_METRICS[name] and m['view'] in requested_views]
        cards.append(dict(factor=name, observations=observations,
            supported_observable_kinds=sorted(FACTOR_OBSERVABLES[name]),
            interpretation='Absence of an alert is not evidence of optimality. Screening thresholds are not calibrated physical acceptance bounds.'))
    return dict(factors=cards, requested_view=view, representations=deepcopy(packet['representations']),
        chemical_readiness=packet['steering']['chemical_readiness'],
        other_view_availability=[{k:deepcopy(m[k]) for k in ('metric_id','view','status')}
                                 for m in packet['observations'] if m['view'] not in {'state','prediction'}],
        representation_rule='Views remain separate current-checkpoint objects. The prediction is a contemporaneous estimate, not a measured terminal structure. Unavailable state chemistry is not evidence of final failure.',
        detail_lookup='Array previews are explicitly bounded. Reference pages contain metadata only, avoiding duplicate whole-metric summaries. Use list_measurement_references for another page or atom filter, then inspect_measurements for exact localized values or complete summaries.',
        categorical_control='Native categories may evolve. This adapter exposes coordinate derivatives only; do not claim direct functional-group induction.',
        missing_target_policy='No supplied contact, area, property or group target means unknown, not a favorable value or permission to invent one.')


def _evidence(packet, report):
    values = {e['evidence_id']: e for m in packet['observations'] for e in m.get('evidence', [])}
    values.update(measurement_references(packet))
    values.update(report['evidence_index'])
    return values


def validate_factor_assessment(biology, packet, report, required=False):
    factors = biology.get('factor_assessment', [])
    if not factors:
        if required:
            raise ContractValidationError('Assess every biophysical factor before selecting a reward', ['factor_assessment'])
        return
    names = [f['factor'] for f in factors]
    if len(names) != len(set(names)) or set(names) != set(FACTORS):
        raise ContractValidationError('Assess every biophysical factor before selecting a reward', ['factor_assessment'])
    directions = {d['direction_id']: d for d in biology['directions']}
    evidence = _evidence(packet, report)
    for index, factor in enumerate(factors):
        path = ['factor_assessment', index]
        if not set(factor['evidence_ids']) <= set(evidence) or not set(factor['direction_ids']) <= set(directions):
            raise ContractValidationError('Factor assessment must bind actual evidence and direction IDs', path)
        if factor['disposition'] in ('optimize', 'constraint'):
            if (not factor['direction_ids'] or not factor['evidence_ids'] or factor['missing_requirements']
                or any(directions[d]['disposition'] != factor['disposition'] for d in factor['direction_ids'])):
                raise ContractValidationError('Active factors need measured evidence and matching controllable directions', path)


def expression_constants(tree):
    if not isinstance(tree, dict):
        return []
    if tree.get('op') == 'constant':
        return [tree]
    return [c for child in tree.get('args', []) for c in expression_constants(child)]


def validate_predicate_claims(direction,path):
    text=direction['repair_predicate'].lower()
    kinds={c['observable_kind'] for c in direction['repair_clauses']}
    # A conservative lint for observed hidden global guarantees. Formal clauses
    # remain authoritative; this does not prove arbitrary prose equivalence.
    required=set()
    if re.search(r'\b(?:no new|zero|all)\b[^.;\n]{0,100}\bclashes?\b',text):
        required.add('whole_clash_screen')
    for token,kind in [('contact_pair_count','contact_count'),('contact_count','contact_count'),
                       ('contacting_ligand_fraction','contact_fraction'),('contact_fraction','contact_fraction'),
                       ('buried_fraction','burial_sasa'),('burial_sasa','burial_sasa')]:
        if token in text:required.add(kind)
    if not required <= kinds:
        raise ContractValidationError('Global repair guarantees must have explicit matching repair clauses',path)


def validate_math_acceptance_claims(direction, biology_direction, path):
    # Formal clauses are authoritative, but reject an observed hidden promise:
    # independent monitors are not installed acceptance gates in the Executor.
    text=direction['acceptable_set'].lower()
    if re.search(r'\bmonitors?\b[^.;\n]{0,240}\bmust\b[^.;\n]{0,100}\b(?:no regression|pass|zero)\b', text):
        raise ContractValidationError('Independent monitors cannot be claimed as executable acceptance gates', path)
    validate_predicate_claims(dict(repair_predicate=direction['acceptable_set'],
        repair_clauses=biology_direction.get('repair_clauses', [])), path)


def _numbers(value):
    if type(value) in (int, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def validate_design_audit(design, biology, packet, report, passages, required=False):
    audit = design.get('design_audit')
    if not audit:
        if required:
            raise ContractValidationError('Reward design requires architecture and parameter audits', ['design_audit'])
        return
    if sum(c['decision'] == 'selected' for c in audit['architectures']) != 1:
        raise ContractValidationError('Select exactly one architecture and explain rejected alternatives', ['design_audit', 'architectures'])
    evidence = _evidence(packet, report)
    chemical_references = {e['evidence_id'] for m in packet['observations']
        if m['metric_id'] in CHEMICAL_REFERENCE_METRICS
        for e in m.get('evidence', [])}
    chemical_references.update(ident for ident, ref in measurement_references(packet).items()
        if ref['metric_id'] in CHEMICAL_REFERENCE_METRICS)
    by_id = {d['direction_id']: d for d in design['directions']}
    for bio in biology['directions']:
        if bio['disposition'] not in ('optimize','constraint'):
            continue
        direction=by_id[bio['direction_id']]
        if direction['status']!='executable':
            continue
        validate_math_acceptance_claims(direction, bio,
            ['directions',design['directions'].index(direction),'acceptable_set'])
        clauses=bio.get('repair_clauses', [])
        coverage=direction.get('predicate_coverage', [])
        path=['directions',design['directions'].index(direction),'predicate_coverage']
        if not clauses and not required:
            continue
        if (not clauses or len({c['clause_id'] for c in coverage})!=len(coverage)
            or {c['clause_id'] for c in coverage}!={c['clause_id'] for c in clauses}):
            raise ContractValidationError('Implement every declared repair clause in the expression',path)
        observables={o['observable_id']:o for o in direction['observables']}
        clause_map={c['clause_id']:c for c in clauses}
        for item in coverage:
            clause=clause_map[item['clause_id']]
            if (not set(item['observable_ids']) <= set(observables) or
                any(observables[ident]['kind']!=clause['observable_kind']
                    for ident in item['observable_ids'])):
                raise ContractValidationError('A repair clause needs its actual observable and bound evidence; no implicit host gates',path)
            # An observable can also cite corroborating measurements used by
            # another clause. This clause's own shared evidence must independently
            # localize the observable; unrelated extra citations cannot satisfy it.
            from molsteer.molthinker.expressions import validate_observables
            for ident in item['observable_ids']:
                bound=deepcopy(observables[ident])
                bound['evidence_ids']=sorted(set(bound['evidence_ids']) & set(clause['evidence_ids']))
                try:
                    validate_observables([bound],packet,set(clause['evidence_ids']))
                except ValueError:
                    raise ContractValidationError('A repair clause needs its actual observable and bound evidence; no implicit host gates',path) from None
    for factor in biology.get('factor_assessment', []):
        if factor['disposition'] not in ('optimize', 'constraint'):
            continue
        implemented = {o['kind'] for ident in factor['direction_ids'] for o in by_id[ident]['observables']}
        if not implemented & FACTOR_OBSERVABLES[factor['factor']]:
            raise ContractValidationError('Reward omits the declared physical repair predicate', ['design_audit', factor['factor']])
    for index, direction in enumerate(design['directions']):
        if direction['status'] != 'executable':
            continue
        path = ['directions', index]
        candidates = direction.get('function_basis', [])
        if not candidates or not any(c['decision'] == 'selected' for c in candidates):
            raise ContractValidationError('Inspect a function basis and identify the implemented source shape', path + ['function_basis'])
        for candidate in candidates:
            if candidate['locator'] not in passages:
                raise ContractValidationError('Function basis must cite inspected source locations', path + ['function_basis'])
        lineage_locators = {lineage['locator'] for lineage in direction['function_lineage']}
        if any(c['locator'] not in lineage_locators for c in candidates if c['decision'] == 'selected'):
            raise ContractValidationError('Function basis must cite inspected source locations', path + ['function_basis'])
        selected_ids = {passages[c['locator']].get('function_id') for c in candidates if c['decision'] == 'selected'}
        kinds = {o['kind'] for o in direction['observables']}
        if 'P01' in selected_ids and 'mmff_strain' not in kinds and direction.get('source_transform') != 'new_surrogate':
            raise ContractValidationError('A distance proxy cannot be attributed to the complete MMFF energy', path + ['source_transform'])
        parameters = direction.get('reference_parameters', [])
        chemical_binding = ('P01' in selected_ids or any(
            p['role'] in ('reference', 'tolerance', 'physical_coefficient') and
            set(p['evidence_ids']) & chemical_references for p in parameters))
        if chemical_binding and audit['graph_policy'] != 'suspend_on_graph_change':
            raise ContractValidationError('Chemical references require a graph-change applicability guard', path + ['reference_parameters'])
        constants = expression_constants(direction['expression'])
        for constant in constants:
            matches = [p for p in parameters if p['origin'] == constant['origin'] and p['unit'] == constant['unit']
                       and math.isclose(p['value'], constant['value'], rel_tol=1e-10, abs_tol=1e-12)]
            if len(matches) != 1:
                raise ContractValidationError('Every expression constant needs one exact parameter provenance record', path + ['reference_parameters'])
        if any(not any(p['origin'] == c['origin'] and p['unit'] == c['unit'] and
                       math.isclose(p['value'], c['value'], rel_tol=1e-10, abs_tol=1e-12) for c in constants) for p in parameters):
            raise ContractValidationError('Parameter provenance cannot describe inactive constants', path + ['reference_parameters'])
        for parameter in parameters:
            if (not set(parameter['evidence_ids']) <= set(evidence) or
                not set(parameter['source_locators']) <= set(passages)):
                raise ContractValidationError('Reference parameters must cite bound evidence or inspected sources', path + ['reference_parameters'])
            provenance = parameter['provenance']
            if provenance == 'observed_reference' and (not parameter['evidence_ids'] or not any(
                math.isclose(parameter['value'], number, rel_tol=1e-6, abs_tol=1e-8)
                for ident in parameter['evidence_ids'] for number in _numbers(evidence[ident]))):
                raise ContractValidationError('Observed reference values must occur in their cited evidence', path + ['reference_parameters'])
            if provenance in ('source_formula', 'calibration') and not parameter['source_locators']:
                raise ContractValidationError('A source-derived parameter needs an inspected formula or calibration', path + ['reference_parameters'])
            if provenance in ('declared_assumption', 'diagnostic_screening') and audit['execution_scope'] != 'bounded_hypothesis_pilot':
                raise ContractValidationError('Screening thresholds and uncalibrated assumptions only authorize a bounded hypothesis pilot', path + ['reference_parameters'])
        _validate_shape_probes(direction, path)


def _validate_shape_probes(direction, path):
    import torch
    from molsteer.molthinker.expressions import evaluate_expression
    probes = direction.get('shape_probes', [])
    ids = {o['observable_id'] for o in direction['observables']}
    if len(probes) < 3:
        raise ContractValidationError('Test the function shape at repaired and violated values', path + ['shape_probes'])
    distinct = set()
    for index, probe in enumerate(probes):
        probe_path = path + ['shape_probes', index]
        if set(probe['observable_values']) != ids or set(probe['derivative_signs']) != ids:
            raise ContractValidationError('Shape probes must account for every retained observable', probe_path)
        distinct.add(tuple(sorted(probe['observable_values'].items())))
        values = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in probe['observable_values'].items()}
        loss = evaluate_expression(direction['expression'], values)
        if (abs(float(loss.detach())) <= 1e-10) != probe['expected_zero']:
            raise ContractValidationError('Executable zero set contradicts the declared shape probe', probe_path + ['expected_zero'])
        gradients = torch.autograd.grad(loss, tuple(values.values()), allow_unused=True)
        for ident, gradient in zip(values, gradients):
            number = 0. if gradient is None else float(gradient)
            sign = 'zero' if abs(number) <= 1e-10 else 'positive' if number > 0 else 'negative'
            if sign != probe['derivative_signs'][ident]:
                raise ContractValidationError('Executable derivative contradicts the declared shape probe', probe_path + ['derivative_signs', ident])
    if len(distinct) < 3 or not any(p['expected_zero'] for p in probes) or all(p['expected_zero'] for p in probes):
        raise ContractValidationError('Test the function shape at repaired and violated values', path + ['shape_probes'])
