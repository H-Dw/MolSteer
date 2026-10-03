"""Read saved raw suffixes without confusing them with current reward inputs.

Comparisons describe sampled evidence, not intervention effects. No sampler,
oracle, LLM or external program is launched by this module.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from jsonschema.exceptions import ValidationError as SchemaValidationError

from molsteer.common import digest
from molsteer.contracts import validate_enriched
from .measurement_refs import measurement_references


LIMITATIONS = [
    'Observed raw continuation is a reference path; guided outcomes remain unobserved.',
    'Sampled clearance brackets a screening change, not its exact time or a causal repair.',
    'Persistence describes selected sampled nodes; unsampled steps are not observed.',
    'Global affinity/SA/strain changes are not a decomposition of regional contributions.',
    'Changed chemical identities or incomplete measurements cannot certify repair of the original defect.',
    'Future reference IDs are contextual citations, never current RewardSpec numerical evidence.',
]
PLANAR_METRICS = {'ring_planarity', 'double_bond_planarity'}
REFERENCE_FIELDS = ('reference_angstrom', 'reference_degrees', 'mmff_parameter_type',
                    'mmff_atom_types', 'bond_order', 'lower_angstrom', 'upper_angstrom',
                    'endpoint_lower_angstrom', 'endpoint_upper_angstrom')


def _base(packet, status, issues=None):
    return dict(kind='RawTrajectoryComparison', schema_version='1.0', status=status,
                anchor_packet_id=packet['packet_id'], anchor_content_hash=digest(packet),
                issues=issues or [], limitations=deepcopy(LIMITATIONS))


def _read(root, relative):
    """Only the configured repository can supply files, including through symlinks."""
    if not isinstance(relative, (str, Path)) or not str(relative).strip():
        raise ValueError('A nonempty repository-relative JSON path is required')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Raw reference paths must stay within the repository')
    root = Path(root).resolve()
    target = (root / path).resolve()
    target.relative_to(root)
    content = target.read_bytes()
    return json.loads(content), dict(path=path.as_posix(), sha256=hashlib.sha256(content).hexdigest())


def _validate_saved(packet):
    if not isinstance(packet, dict):
        raise ValueError('Saved StatePacket must be an object')
    try:
        validate_enriched(packet)
    except SchemaValidationError:
        raise ValueError('Saved StatePacket fails its schema') from None


def load_raw_reference(packet, config, repo_root):
    """Load only selected saved nodes; unavailable reference input is advisory.

    The manifest's raw trajectory claim is declared provenance. Exact anchor
    content binding and per-node file hashes are verified by the host.
    """
    validate_enriched(packet)
    if not config.enabled:
        return _base(packet, 'disabled')
    try:
        manifest, source = _read(repo_root, config.manifest_path)
        if (not isinstance(manifest, dict) or manifest.get('kind') != 'RawInferenceManifest' or manifest.get('schema_version') != '1.0'
                or manifest.get('mode') != 'raw' or not isinstance(manifest.get('trajectory_id'), str)
                or not manifest['trajectory_id'].strip()
                or manifest.get('time_direction') not in ('increasing', 'decreasing')):
            raise ValueError('Use a versioned raw manifest with trajectory_id and explicit time_direction')
        anchor, anchor_source = _read(repo_root, manifest.get('anchor_packet_path'))
        _validate_saved(anchor)
        if digest(anchor) != digest(packet):
            raise ValueError('The configured raw anchor does not match the current packet content')
        anchor_t = packet['identity']['stage_t']
        if not _number(anchor_t):
            raise ValueError('Current packet must declare a finite stage_t')
        declarations = manifest.get('nodes')
        if not isinstance(declarations, list) or not declarations:
            raise ValueError('Raw manifest requires saved node declarations')
        node_ids, times = set(), set()
        for node in declarations:
            if (not isinstance(node, dict) or not isinstance(node.get('node_id'), str)
                    or not node['node_id'].strip() or node['node_id'] == 'anchor'
                    or not _number(node.get('time'))
                    or node.get('role') not in ('intermediate', 'final')
                    or node['node_id'] in node_ids or node['time'] in times):
                raise ValueError('Declare distinct node IDs/times and explicit intermediate/final roles')
            node_ids.add(node['node_id']); times.add(node['time'])
        sign = 1 if manifest['time_direction'] == 'increasing' else -1
        finals = [n for n in declarations if n['role'] == 'final']
        if len(finals) > 1 or (finals and any(sign * (n['time'] - finals[0]['time']) > 0 for n in declarations)):
            raise ValueError('A single explicit final must be the terminal declared node')
        selected = [n for n in declarations if n['time'] in config.reference_times
                    or (config.include_final and n['role'] == 'final')]
        issues = [dict(kind='requested_time_not_saved', time=t) for t in config.reference_times if t not in times]
        selected += [dict(node_id='missing_'+digest(t)[:12], time=t, role='intermediate', packet_path=None)
                     for t in config.reference_times if t not in times]
        if config.include_final and not finals:
            issues.append(dict(kind='final_not_declared'))
        nodes = [dict(node_id='anchor', role='anchor', time=anchor_t, packet=anchor, source=anchor_source)]
        for declared in sorted(selected, key=lambda n: sign * n['time']):
            node = dict(node_id=declared['node_id'], role=declared['role'], time=declared['time'], packet=None)
            try:
                if sign * (declared['time'] - anchor_t) <= 0:
                    raise ValueError('Reference nodes must follow the current anchor in the declared time direction')
                later, node_source = _read(repo_root, declared.get('packet_path'))
                _validate_saved(later)
                if later['identity']['stage_t'] != declared['time']:
                    raise ValueError('Node time does not match its packet')
                if any(later['identity'][key] != anchor['identity'][key] for key in ('target_id', 'ligand_id')):
                    raise ValueError('Reference node belongs to a different target or ligand')
                node.update(packet=later, source=node_source)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                node['issue'] = str(exc)
                issues.append(dict(kind='node_unavailable', node_id=node['node_id'], reason=str(exc)))
            nodes.append(node)
        comparison = compare_raw_nodes(packet, nodes, config)
        comparison.update(trajectory_id=manifest['trajectory_id'], time_direction=manifest['time_direction'],
                          manifest_source=source, provenance=deepcopy(manifest.get('provenance', {})),
                          provenance_status='declared_raw_path_with_verified_anchor_content', issues=issues)
        if len(nodes) == 1 or not any(n['packet'] for n in nodes[1:]):
            comparison['status'] = 'unavailable'
        elif issues:
            comparison['status'] = 'partial'
        comparison['comparison_id'] = 'rtc_' + digest(comparison)[:24]
        return comparison
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _base(packet, 'unavailable', [dict(kind='reference_unavailable', reason=str(exc))])


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _atoms(value):
    atoms = value.get('atom_ids', [])
    return tuple(atoms) if atoms else ((value['atom_id'],) if type(value.get('atom_id')) is int else ())


def _location(value, unordered=False):
    atoms = _atoms(value)
    # Reverse a bond/angle/torsion without losing the central atom order.
    atoms = tuple(sorted(atoms)) if unordered else min(atoms, atoms[::-1])
    return atoms, tuple((key, str(value[key])) for key in
        ('receptor_serial', 'residue_id', 'check', 'kind', 'center_atom_id') if key in value)


def _risk_key(metric, evidence):
    # These are molecule-level measurements, not an energy decomposition.
    global_metric = metric['metric_id'] in ('mmff_strain', 'mmff_energy', 'posebusters')
    location = _location({**evidence, 'atom_ids': [], 'atom_id': None}) if global_metric else _location(evidence, metric['metric_id'] in PLANAR_METRICS)
    return metric['metric_id'], metric['view'], location, evidence.get('message')


def _chemical_signature(packet, view, atoms):
    metric = next((m for m in packet['observations'] if m['metric_id'] == 'chemistry_context'
                   and m['view'] == view and m['status'] in ('ok', 'partial')), None)
    if not metric:
        return None
    values = metric['values']
    chemical_atoms = {r['atom_id']: r for r in values.get('atoms', [])}
    selected = atoms or tuple(sorted(chemical_atoms))
    rows = []
    for atom in selected:
        row = chemical_atoms.get(atom)
        if not row or row.get('element') in (None, 'PAD', 'unknown') or row.get('formal_charge') is None:
            return None
        neighbors = []
        for bond in values.get('bonds', []):
            if atom in bond['atom_ids'] and bond.get('bond_order', 0) > 0:
                other = next(a for a in bond['atom_ids'] if a != atom)
                neighbor = chemical_atoms.get(other, {})
                if neighbor.get('element') in (None, 'PAD', 'unknown') or neighbor.get('formal_charge') is None:
                    return None
                neighbors.append((other, bond['bond_order'], neighbor.get('element'), neighbor.get('formal_charge')))
        rows.append((atom, row['element'], row['formal_charge'], row.get('hybridization'), sorted(neighbors)))
    return digest(rows)


def _compatible(anchor, later, metric, other, atoms, facts):
    if other is None or other['status'] not in ('ok', 'partial'):
        return 'measurement_unavailable'
    if any(metric.get(k) != other.get(k) for k in ('method', 'units', 'thresholds', 'metric_version', 'config_hash')):
        return 'measurement_definition_changed'
    view = metric['view']
    a, b = anchor['representations'].get(view, {}), later['representations'].get(view, {})
    if any(a.get(k) != b.get(k) for k in ('coordinate_frame', 'coordinate_unit')):
        return 'representation_frame_changed'
    if not atoms and anchor['steering']['graph_signatures'].get(view) != later['steering']['graph_signatures'].get(view):
        return 'chemical_identity_changed'
    receptor_metrics = {'protein_contacts', 'protein_clashes', 'hydrogen_bond_candidates',
        'hydrophobic_contacts', 'aromatic_contacts', 'salt_bridge_candidates', 'burial_sasa'}
    if metric['metric_id'] in receptor_metrics:
        receptors = [facts.setdefault(('receptor', id(p)), {}) for p in (anchor, later)]
        for p, cached in zip((anchor, later), receptors):
            if 'hash' not in cached:
                cached['hash'] = digest(p['steering']['receptor_atoms']) if p['steering']['receptor_atoms'] else None
        if receptors[0]['hash'] is None or receptors[0]['hash'] != receptors[1]['hash']:
            return 'receptor_context_unverified'
    signatures = []
    for p in (anchor, later):
        key = ('chemistry', id(p), view, atoms)
        if key not in facts:
            facts[key] = _chemical_signature(p, view, atoms)
        signatures.append(facts[key])
    if None in signatures:
        return 'chemical_assignment_unavailable'
    if signatures[0] != signatures[1]:
        return 'chemical_identity_changed'
    return 'comparable'


def _metric_rows(refs, metric):
    return [r for r in refs.values() if r['view'] == metric['view'] and r['metric_id'] == metric['metric_id']
            and r['kind'] in ('measured_row', 'derived_geometry')]


def _supports(row, evidence):
    a, b = _atoms(row['values']), _atoms(evidence)
    unordered = row['metric_id'] in PLANAR_METRICS
    if (tuple(sorted(a)) if unordered else min(a, a[::-1])) != (tuple(sorted(b)) if unordered else min(b, b[::-1])):
        return False
    return all(row['values'].get(k) == evidence[k] for k in ('receptor_serial', 'check', 'kind')
               if k in evidence and k in row['values'])


def _local_reference_changed(evidence, origin_rows, later_rows, later_flags):
    original = [r['values'] for r in origin_rows] or [evidence]
    later = [r['values'] for r in later_rows] or later_flags
    for values in original:
        parameters = {k: values[k] for k in REFERENCE_FIELDS if k in values}
        if parameters and later and not any(all(k in row and row[k] == v for k, v in parameters.items()) for row in later):
            return True
    return False


def _clearance_covered(metric, refs, evidence):
    """Absence of a flag is not sufficient: require evaluated local coverage."""
    matches = [r for r in _metric_rows(refs, metric) if _supports(r, evidence)]
    if matches:
        # A partially measured metric can still have a fully evaluated local row.
        return any(all(r['values'].get(k) is not None for k in evidence
                       if _number(evidence[k]) and k not in ('receptor_serial',)) for r in matches)
    if metric['status'] != 'ok':
        return False
    name, values = metric['metric_id'], metric['values']
    if name == 'protein_clashes':
        return values.get('unknown_ligand_atoms') == 0 and values.get('checked_pair_count', 0) > 0
    if name == 'intramolecular_clashes':
        return values.get('unknown_element_pair_count') == 0 and values.get('checked_pair_count', 0) > 0
    if name in ('mmff_strain', 'mmff_energy'):
        return bool(values) and values.get('minimization_status', 0) == 0
    if name == 'posebusters':
        return evidence.get('check') in values.get('checks', {})
    return not _atoms(evidence) and bool(values)


def _current_condition(metric, refs, evidence, local_flags, chemical, atom_ids):
    """Interpret a node's own complete screens, without requiring old references.

    Removed relations are inapplicable rather than repaired. Complete valence or
    probability screens can clear a local alert even when no outlier row remains.
    """
    if not metric or metric['status'] not in ('ok', 'partial'):
        return 'unavailable'
    if local_flags:
        return 'flagged'
    if _clearance_covered(metric, refs, evidence):
        return 'not_flagged'
    name, values, atoms = metric['metric_id'], metric['values'], _atoms(evidence)
    if metric['status'] == 'ok':
        if name == 'valence' and values.get('sanitized') is True and set(atoms) <= {
                r['atom_id'] for r in values.get('explicit_bond_order_sum', [])}:
            return 'not_flagged'
        if name in ('atom_confidence', 'charge_confidence', 'bond_confidence'):
            n = len(atom_ids)
            expected = n*(n-1)//2 if name == 'bond_confidence' else n
            if (set(atoms) <= set(atom_ids) and values.get('row_count') == expected and
                    values.get('low_confidence_count') == len(metric.get('evidence', []))):
                return 'not_flagged'
    bonded = (name == 'bond_lengths' and len(atoms) == 2 or name == 'mmff_local_geometry'
              and evidence.get('kind') in ('bond_length', 'bond_angle'))
    if bonded and chemical.get('status') == 'ok':
        chem = chemical['values']
        if set(atoms) <= {a['atom_id'] for a in chem.get('atoms', [])}:
            edges = {frozenset(b['atom_ids']) for b in chem.get('bonds', []) if b.get('bond_order', 0) > 0}
            if any(frozenset(pair) not in edges for pair in zip(atoms, atoms[1:])):
                return 'relation_absent'
    return 'local_coverage_unavailable'


def _value_path(values, path):
    for key in path:
        if not isinstance(values, dict) or key not in values:
            return None
        values = values[key]
    return values if _number(values) else None


def _numeric_changes(before, after):
    # Preserve source units and raw values; do not label every decreasing distance helpful.
    return {key: dict(before=before[key], after=after[key], delta=after[key]-before[key])
            for key in before.keys() & after.keys()
            if _number(before[key]) and _number(after[key]) and before[key] != after[key]
            and not key.endswith('_id') and key not in ('atom_id', 'receptor_serial', 'mmff_parameter_type')}


def _outcome_status(anchor, later, metric, other, view, path):
    if metric is None or other is None or metric['status'] not in ('ok', 'partial') or other['status'] not in ('ok', 'partial'):
        return 'measurement_unavailable'
    if any(metric[k] != other[k] for k in ('method', 'units', 'metric_version', 'config_hash')):
        return 'measurement_definition_changed'
    if metric['metric_id'] in ('affinity', 'vina_score', 'vinardo_score'):
        a, b = anchor['representations'].get(view, {}), later['representations'].get(view, {})
        if any(a.get(k) != b.get(k) for k in ('coordinate_frame', 'coordinate_unit')):
            return 'representation_frame_changed'
        if not anchor['steering']['receptor_atoms'] or digest(anchor['steering']['receptor_atoms']) != digest(later['steering']['receptor_atoms']):
            return 'receptor_context_unverified'
    if metric['metric_id'] in ('mmff_energy', 'mmff_strain') and path[-1] in ('energy_kcal_mol', 'relaxed_energy_kcal_mol'):
        if anchor['steering']['graph_signatures'].get(view) != later['steering']['graph_signatures'].get(view):
            return 'absolute_energy_across_chemical_identities'
    return 'comparable'


def compare_raw_nodes(packet, nodes, config):
    """Compare host-validated, ordered saved nodes. Selection is supplied by config."""
    from molsteer.agents.design_audit import FACTOR_METRICS
    context = _base(packet, 'available')
    selected_factors = config.factors or list(FACTOR_METRICS)
    factor_of = {metric: [f for f in selected_factors if metric in FACTOR_METRICS[f]]
                 for metric in set().union(*(FACTOR_METRICS[f] for f in selected_factors))}
    context.update(nodes=[], reference_index={}, risk_tracks=[], regional_changes=[], node_diagnostics=[],
                   outcome_trends=[], opportunities=[], factor_coverage={f: [] for f in selected_factors},
                   selection=config.model_dump(mode='json'))
    cache, facts = {}, {}

    def reference(node, record, kind=None):
        payload = dict(node_id=node['node_id'], time=node['time'], packet_id=node['packet']['packet_id'],
                       packet_content_hash=cache[node['node_id']]['content_hash'], source=node.get('source'),
                       record=deepcopy(record), role='current' if node['role'] == 'anchor' else 'raw_reference',
                       kind=kind or record.get('kind', 'metric_evidence'))
        ident = 'rr_' + digest(payload)[:24]
        context['reference_index'][ident] = payload
        return ident

    for node in nodes:
        later = node['packet']
        public = {k: deepcopy(v) for k, v in node.items() if k != 'packet'}
        public['status'] = 'available' if later else 'unavailable'
        if later:
            metrics = {(m['metric_id'], m['view']): m for m in later['observations'] if m['view'] in config.views}
            refs = measurement_references(later)
            cache[node['node_id']] = dict(metrics=metrics, refs=refs, content_hash=digest(later))
            public.update(packet_id=later['packet_id'], content_hash=cache[node['node_id']]['content_hash'],
                          representations=deepcopy(later['representations']),
                          graph_signatures=deepcopy(later['steering']['graph_signatures']))
            public['metric_references'] = []
            for (name, view), metric in metrics.items():
                summary = next((r for r in refs.values() if r['metric_id'] == name and r['view'] == view and not r['path']), None)
                ref_id = reference(node, summary or metric, 'metric_summary')
                public['metric_references'].append(dict(metric_id=name, view=view, status=metric['status'], reference_id=ref_id))
                for factor in factor_of.get(name, []):
                    context['factor_coverage'][factor].append(dict(node_id=node['node_id'], metric_id=name, view=view,
                                                                  status=metric['status'], reference_id=ref_id))
            # Diagnose each saved node on its OWN chemistry. Origin comparability
            # below remains separate from whether the later molecule is defective.
            for view in config.views:
                for factor in selected_factors:
                    panel = [m for (name, v), m in metrics.items() if v == view and factor in factor_of.get(name, [])]
                    context['node_diagnostics'].append(dict(node_id=node['node_id'], time=node['time'],
                        view=view, factors=[factor], factor=factor,
                        measured_metrics=[m['metric_id'] for m in panel if m['status'] in ('ok', 'partial')],
                        unavailable_metrics=[m['metric_id'] for m in panel if m['status'] not in ('ok', 'partial')],
                        flagged_evidence_count=sum(len(m['evidence']) for m in panel if m['status'] in ('ok', 'partial'))))
        context['nodes'].append(public)

    anchor = nodes[0]
    anchor_cache = cache[anchor['node_id']]
    # Union also exposes late-emerging risks; they are not invented current findings.
    risks = {}
    for node in nodes:
        for metric in cache.get(node['node_id'], {}).get('metrics', {}).values():
            if metric['metric_id'] in factor_of and metric['status'] in ('ok', 'partial'):
                for evidence in metric['evidence']:
                    risks.setdefault(_risk_key(metric, evidence), (node, metric, evidence))
    for key, (origin, basis, evidence) in risks.items():
        name, view = key[:2]
        atoms = () if name in ('mmff_strain', 'mmff_energy', 'posebusters') else _atoms(evidence)
        track = dict(track_id='rt_' + digest((packet['packet_id'], key))[:24], metric_id=name, view=view,
                     factors=factor_of[name], atom_ids=list(atoms), origin_node_id=origin['node_id'],
                     origin_reference_id=reference(origin, evidence), observations=[])
        origin_rows = [r for r in _metric_rows(cache[origin['node_id']]['refs'], basis) if _supports(r, evidence)]
        for node in nodes:
            cached = cache.get(node['node_id'], {})
            metric = cached.get('metrics', {}).get((name, view))
            flags = [e for e in (metric or {}).get('evidence', []) if _risk_key(metric, e) == key]
            supporting = [r for r in _metric_rows(cached['refs'], metric) if _supports(r, evidence)] if metric else []
            status = _compatible(origin['packet'], node['packet'], basis, metric, atoms, facts) if node['packet'] else 'measurement_unavailable'
            if status == 'comparable' and _local_reference_changed(evidence, origin_rows, supporting, flags):
                status = 'local_reference_changed_or_unavailable'
            if status == 'comparable':
                status = 'flagged' if flags else ('not_flagged' if _clearance_covered(metric, cached['refs'], evidence) else 'local_coverage_unavailable')
            obs = dict(node_id=node['node_id'], time=node['time'], status=status, observed_flag=bool(flags),
                       reference_ids=[reference(node, e) for e in flags])
            local_flags = [e for e in (metric or {}).get('evidence', []) if
                (not atoms or _location(e, name in PLANAR_METRICS) == _location(evidence, name in PLANAR_METRICS))]
            chemical = cached.get('metrics', {}).get(('chemistry_context', view), {})
            ids = node['packet']['representations'].get(view, {}).get('original_atom_ids', []) if node['packet'] else []
            obs['current_condition_status'] = _current_condition(metric, cached.get('refs', {}), evidence, local_flags, chemical, ids)
            obs['current_condition_reference_ids'] = [reference(node, e) for e in local_flags]
            obs['current_reference_parameters'] = [{k: deepcopy(r['values'][k]) for k in REFERENCE_FIELDS if k in r['values']}
                                                   for r in supporting]
            if node['packet']:
                chem = cached.get('metrics', {}).get(('chemistry_context', view), {}).get('values', {})
                obs['current_atom_types'] = [{k: a.get(k) for k in ('atom_id', 'element', 'formal_charge', 'hybridization')}
                                            for a in chem.get('atoms', []) if a['atom_id'] in atoms]
            if metric:
                obs['measurement_reference_ids'] = [reference(node, r) for r in supporting]
                if node['role'] == 'anchor':
                    track['current_evidence_ids'] = [e['evidence_id'] for e in flags]
                    track['current_measurement_ids'] = [r['evidence_id'] for r in supporting]
            track['observations'].append(obs)
        statuses = [o['status'] for o in track['observations']]
        final = next((i for i, n in enumerate(nodes) if n['role'] == 'final'), None)
        clear = next((i for i, s in enumerate(statuses) if i and s == 'not_flagged'), None)
        last_flagged = max((i for i, s in enumerate(statuses[:clear]) if s == 'flagged'), default=None) if clear else None
        track['first_observed_clearance'] = (dict(last_flagged_node=nodes[last_flagged]['node_id'] if last_flagged is not None else None,
            first_not_flagged_node=nodes[clear]['node_id'], interval=[nodes[last_flagged]['time'], nodes[clear]['time']] if last_flagged is not None else None,
            selected_measurement_gaps_between=any(s != 'flagged' for s in statuses[last_flagged+1:clear]) if last_flagged is not None else True,
            unsampled_steps='not_observed') if clear else None)
        if any(s == 'chemical_identity_changed' for s in statuses):
            classification = 'chemical_retyped_or_graph_changed'
        elif statuses[0] != 'flagged':
            classification = 'emerged_in_sampled_suffix' if statuses[0] == 'not_flagged' else 'initial_condition_unresolved'
        elif final is not None and statuses[final] == 'flagged':
            classification = 'persistent_through_raw_final' if all(s == 'flagged' for s in statuses[:final+1]) else 'flagged_at_raw_final_with_gaps_or_recurrence'
        elif final is not None and statuses[final] == 'not_flagged' and clear:
            classification = 'first_clearance_at_raw_final' if clear == final else 'cleared_before_raw_final'
        elif clear:
            classification = 'clearance_observed_in_suffix'
        else:
            classification = 'persistence_observed_in_suffix' if all(s == 'flagged' for s in statuses) else 'unresolved_measurement_or_definition_change'
        track['classification'] = classification
        track['current_condition_trajectory'] = _condition_trajectory(track['observations'], nodes)
        context['risk_tracks'].append(track)
        if 'persistent_defects' in config.analyses and classification in ('persistent_through_raw_final', 'flagged_at_raw_final_with_gaps_or_recurrence'):
            context['opportunities'].append(_opportunity('additional_repair_assessment', track,
                'Assess whether this current risk needs extra repair; raw final still flags it. Check coverage, chemistry and coupled risks.'))
        if 'late_repair' in config.analyses and statuses[0] == 'flagged' and clear and classification != 'chemical_retyped_or_graph_changed':
            context['opportunities'].append(_opportunity('earlier_repair_assessment', track,
                'Assess whether earlier intervention helps or disrupts a raw repair already observed. Timing is bracketed by sampled nodes; recurrence and unknown intervals remain visible.'))
        if classification == 'chemical_retyped_or_graph_changed':
            context['opportunities'].append(_opportunity('chemical_transition_assessment', track,
                'The original chemical hypothesis changed. Inspect current_condition_trajectory and typed references: '
                'does a defect remain in the later chemistry, or would enforcing the old target oppose native evolution?'))

    # Global trends: configuration supplies meaning/units of a favorable direction.
    improved_nodes = {}
    for selection in config.outcome_metrics:
        base = anchor_cache['metrics'].get((selection.metric_id, selection.view))
        start = _value_path(base['values'], selection.value_path) if base and base['status'] in ('ok', 'partial') else None
        trend = dict(metric_id=selection.metric_id, value_path=selection.value_path, view=selection.view,
                     direction=selection.direction, min_delta=selection.min_delta, observations=[])
        for node in nodes:
            cached = cache.get(node['node_id'], {})
            metric = cached.get('metrics', {}).get((selection.metric_id, selection.view))
            value = _value_path(metric['values'], selection.value_path) if metric and metric['status'] in ('ok', 'partial') else None
            comparison_status = _outcome_status(packet, node['packet'], base, metric, selection.view, selection.value_path)
            delta = value-start if comparison_status == 'comparable' and value is not None and start is not None else None
            improvement = delta is not None and ((delta if selection.direction == 'higher' else -delta) > selection.min_delta)
            row = dict(node_id=node['node_id'], time=node['time'], value=value, delta_from_anchor=delta,
                       comparison_status=comparison_status, measurement_status=metric['status'] if metric else 'unavailable',
                       directionally_improved_by_config=improvement, units=deepcopy(metric['units']) if metric else {},
                       chemical_graph_changed=bool(node['packet'] and node['packet']['steering']['graph_signatures'].get(selection.view) != packet['steering']['graph_signatures'].get(selection.view)))
            if metric:
                row['reference_id'] = reference(node, metric, 'outcome_metric')
            trend['observations'].append(row)
            if improvement:
                improved_nodes.setdefault(node['node_id'], []).append(dict(metric_id=selection.metric_id, value_path=selection.value_path,
                                                                           view=selection.view,
                                                                           reference_id=row['reference_id'], delta=delta))
        trend['attribution'] = 'molecule_level_proxy_or_measurement; no regional decomposition'
        context['outcome_trends'].append(trend)

    # Regional feature evolution, including absent-at-anchor contacts and identity changes.
    rows = {}
    for node in nodes:
        cached = cache.get(node['node_id'], {})
        for metric in cached.get('metrics', {}).values():
            if metric['metric_id'] not in factor_of:
                continue
            for record in _metric_rows(cached['refs'], metric):
                if metric['metric_id'] == 'chemistry_context' and record['path'][0] == 'bonds' and record['values'].get('bond_order') == 0:
                    continue
                row_key = metric['metric_id'], metric['view'], record['path'][0], _location(record['values'], metric['metric_id'] in PLANAR_METRICS)
                rows.setdefault(row_key, {})[node['node_id']] = (metric, record)
    for key, entries in rows.items():
        name, view, row_group, location = key
        atoms = list(location[0])
        baseline = entries.get('anchor')
        base_metric = anchor_cache['metrics'].get((name, view))
        observations, changed, associated = [], False, []
        for node in nodes:
            entry = entries.get(node['node_id'])
            metric = cache.get(node['node_id'], {}).get('metrics', {}).get((name, view))
            row = dict(node_id=node['node_id'], time=node['time'], listed=bool(entry),
                       coverage='available_screen' if metric and metric['status'] in ('ok', 'partial') else 'unavailable')
            if entry:
                row.update(reference_id=reference(node, entry[1]),
                           current_measurement_id=entry[1]['evidence_id'] if node['role'] == 'anchor' else None)
                row['comparison_status'] = _compatible(packet, node['packet'], base_metric, metric, tuple(atoms), facts) if base_metric else 'anchor_measurement_unavailable'
                row['numeric_changes'] = _numeric_changes(baseline[1]['values'], entry[1]['values']) if baseline else {}
                if row_group == 'atoms' and name == 'chemistry_context' and baseline:
                    row['identity_changes'] = {k: dict(before=baseline[1]['values'].get(k), after=entry[1]['values'].get(k))
                        for k in ('element', 'formal_charge', 'hybridization') if baseline[1]['values'].get(k) != entry[1]['values'].get(k)}
                local_change = bool(row['numeric_changes'] or row.get('identity_changes') or
                                    (not baseline and node['role'] != 'anchor'))
                changed |= local_change
                if local_change and node['node_id'] in improved_nodes:
                    outcomes = [r for r in improved_nodes[node['node_id']] if r['view'] == view or config.cross_view_associations]
                    if outcomes:
                        associated.append(dict(node_id=node['node_id'], regional_reference_id=row['reference_id'],
                            representation_relation='same_view' if all(r['view'] == view for r in outcomes) else 'cross_view_coevolution',
                            outcome_changes=outcomes))
            elif baseline and row['coverage'] == 'available_screen':
                changed = True
            observations.append(row)
        if not changed:
            continue
        change = dict(change_id='rf_' + digest((packet['packet_id'], key))[:24], metric_id=name, view=view,
                      row_group=row_group, atom_ids=atoms, factors=factor_of[name], observations=observations,
                      current_measurement_ids=[baseline[1]['evidence_id']] if baseline else [],
                      interpretation='Observed local evolution; not-listed means absent from this available screen, not proven physical absence.')
        context['regional_changes'].append(change)
        if 'beneficial_regions' in config.analyses and associated:
            opportunity = _opportunity('regional_enhancement_assessment', change,
                'Local changes coincide with configured global improvements. Assess their mechanism, SA/affinity/strain tradeoffs and current controllability; association alone does not establish benefit or justify a reward.')
            opportunity['associations'] = associated
            opportunity['evidence_class'] = 'raw_path_association'
            context['opportunities'].append(opportunity)
    if config.group_regional_candidates:
        context['opportunities'] = _group_regional_opportunities(context['opportunities'])
    from .residual_needs import build_residual_needs
    context['residual_needs'] = build_residual_needs(context)
    return context


def _condition_trajectory(observations, nodes):
    """Describe actual per-node screens without calling retyping an old-object repair."""
    states = [o['current_condition_status'] for o in observations]
    terminal = next((i for i, n in enumerate(nodes) if n['role'] == 'final'), None)
    first_flag = next((i for i, s in enumerate(states) if s == 'flagged'), None)
    clear = next((i for i, s in enumerate(states) if first_flag is not None and i > first_flag and s == 'not_flagged'), None)
    return dict(anchor_status=states[0], final_status=states[terminal] if terminal is not None else 'not_observed',
        final_node_id=nodes[terminal]['node_id'] if terminal is not None else None,
        sampled_flagged_nodes=[o['node_id'] for o in observations if o['current_condition_status'] == 'flagged'],
        sampled_clear_nodes=[o['node_id'] for o in observations if o['current_condition_status'] == 'not_flagged'],
        inapplicable_relation_nodes=[o['node_id'] for o in observations if o['current_condition_status'] == 'relation_absent'],
        first_clear_node_id=nodes[clear]['node_id'] if clear is not None else None,
        recurrent_after_clearance=bool(clear is not None and 'flagged' in states[clear+1:]),
        unknown_nodes=[o['node_id'] for o in observations if o['current_condition_status'] not in ('flagged', 'not_flagged', 'relation_absent')],
        chemical_transition_nodes=[o['node_id'] for o in observations if o['status'] == 'chemical_identity_changed'],
        interpretation='Node-local diagnosis follows that node chemical interpretation; clearance after retyping is not repair of the original chemical object.')


def inspect_goal_trajectory(context, *, atom_ids=None, evidence_ids=None, factor=None, offset=0, limit=8):
    """Localize all selected times for a proposed mechanism; do not rank by alert count."""
    atoms, evidence = set(atom_ids or []), set(evidence_ids or [])
    tracks = [r for r in context.get('risk_tracks', []) if
        (not atoms or atoms.intersection(r['atom_ids'])) and
        (not evidence or evidence.intersection(r.get('current_evidence_ids', []) + r.get('current_measurement_ids', []))) and
        (factor is None or factor in r['factors'])]
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 32:
        return dict(status='needs_input', blocking=False, hint='Use a nonnegative offset and a limit between 1 and 32.')
    return dict(status=context['status'], comparison_id=context.get('comparison_id'),
        tracks=deepcopy(tracks[offset:offset+limit]), total_count=len(tracks),
        next_offset=offset+limit if offset+limit < len(tracks) else None,
        outcome_trends=deepcopy(context.get('outcome_trends', [])),
        regional_candidate_ids=[r['opportunity_id'] for r in context.get('opportunities', [])
            if r['category'] == 'regional_enhancement_assessment' and (not atoms or atoms.intersection(r['atom_ids']))],
        interpretation='Raw outcomes are observed; intervention value is an expert hypothesis requiring mechanism, controllability and tradeoff reasoning.')


def _group_regional_opportunities(opportunities):
    """Collate identical atom support for display, without claiming a shared cause."""
    output, regions = [], {}
    for item in opportunities:
        if item['category'] != 'regional_enhancement_assessment':
            output.append(item)
            continue
        key = item['view'], tuple(sorted(item['atom_ids']))
        if key not in regions:
            region = {k: deepcopy(v) for k, v in item.items() if k not in
                ('source_record_id', 'metric_id', 'associations', 'factors', 'current_measurement_ids', 'current_evidence_ids')}
            region.update(opportunity_id='ro_' + digest(('region', key, item['opportunity_id']))[:24],
                source_record_ids=[], metric_ids=[], associations=[], factors=[], current_measurement_ids=[], current_evidence_ids=[],
                grouping='Identical atom-support display only; shared support does not imply a common mechanism or one sufficient intervention.')
            regions[key] = region
            output.append(region)
        region = regions[key]
        region['source_record_ids'].append(item['source_record_id'])
        region['associations'].extend(dict(source_record_id=item['source_record_id'], **a) for a in item['associations'])
        for field, values in (('factors', item['factors']), ('metric_ids', [item['metric_id']]),
                              ('current_measurement_ids', item['current_measurement_ids']),
                              ('current_evidence_ids', item['current_evidence_ids'])):
            region[field] = list(dict.fromkeys([*region[field], *values]))
    return output


def _opportunity(kind, record, question):
    source_id = record.get('track_id', record.get('change_id'))
    return dict(opportunity_id='ro_' + digest((kind, source_id))[:24], category=kind,
                source_record_id=source_id, metric_id=record['metric_id'], view=record['view'],
                factors=record['factors'], atom_ids=record['atom_ids'], decision_question=question,
                current_evidence_ids=record.get('current_evidence_ids', []),
                current_measurement_ids=record.get('current_measurement_ids', []),
                auto_selected=False, intervention_effect='unobserved')


def raw_reference_summary(context):
    """Small prompt payload; complete facts are available through paged tools."""
    result = {k: deepcopy(context[k]) for k in ('kind', 'status', 'comparison_id', 'anchor_packet_id',
        'anchor_content_hash', 'trajectory_id', 'time_direction', 'provenance_status', 'issues', 'limitations') if k in context}
    result['nodes'] = [{k: deepcopy(n[k]) for k in ('node_id', 'time', 'role', 'status', 'packet_id') if k in n}
                       for n in context.get('nodes', [])]
    result['risk_classifications'] = dict(Counter(r['classification'] for r in context.get('risk_tracks', [])))
    result['opportunity_categories'] = dict(Counter(r['category'] for r in context.get('opportunities', [])))
    result['counts'] = {key: len(context.get(key, [])) for key in ('risk_tracks', 'regional_changes', 'outcome_trends', 'opportunities')}
    result['residual_need_states'] = dict(Counter(r['evolution'] for r in context.get('residual_needs', [])))
    result['residual_need_count'] = len(context.get('residual_needs', []))
    result['decision_order'] = 'Page residual_needs first: explicit terminal remainder, current precursor, independent coverage, preservation and uncertainty. Chemistry transition is separate from defect persistence.'
    result['current_final_statuses'] = dict(Counter(r.get('current_condition_trajectory', {}).get('final_status', 'unknown')
                                                   for r in context.get('risk_tracks', [])))
    from molsteer.agents.design_audit import bounded_values
    result['opportunity_previews'] = {category: [bounded_values(r, 3) for r in context.get('opportunities', []) if r['category'] == category][:3]
                                     for category in result['opportunity_categories']}
    result['preview_scope'] = 'Source-order bounded examples, not a ranking or scientific selection; use the paged tools for full records.'
    result['outcome_trends'] = deepcopy(context.get('outcome_trends', []))
    result['tools'] = 'inspect_raw_goal_trajectory localizes all configured nodes for a candidate; inspect_raw_comparison pages facts; read_raw_reference reads exact evidence. Current measurement IDs remain separate.'
    return result


def bound_raw_context(context, packet):
    """A new checkpoint must not reuse a previous checkpoint's temporal claims."""
    if context and context.get('anchor_content_hash') == digest(packet):
        return context
    return _base(packet, 'unavailable' if context else 'disabled',
                 [dict(kind='stale_anchor', reason='Raw comparison belongs to a different current checkpoint')] if context else [])


def inspect_comparison(context, section='summary', factor=None, view=None, offset=0, limit=8):
    if section == 'summary':
        return raw_reference_summary(context)
    if section not in ('residual_needs', 'nodes', 'node_diagnostics', 'risk_tracks', 'regional_changes', 'outcome_trends', 'opportunities', 'factor_coverage'):
        return dict(status='needs_input', blocking=False, sections=['summary', 'residual_needs', 'nodes', 'node_diagnostics', 'risk_tracks', 'regional_changes', 'outcome_trends', 'opportunities', 'factor_coverage'])
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 32:
        return dict(status='needs_input', blocking=False, hint='Use nonnegative offset and limit 1-32')
    if section == 'factor_coverage':
        records = [dict(factor=f, observations=deepcopy(rows)) for f, rows in context.get(section, {}).items() if factor is None or f == factor]
    else:
        records = [r for r in context.get(section, []) if (factor is None or factor in r.get('factors', []))
                   and (view is None or view == r.get('view'))]
    return dict(status=context['status'], comparison_id=context.get('comparison_id'), section=section,
                records=deepcopy(records[offset:offset+limit]), total_count=len(records), offset=offset,
                next_offset=offset+limit if offset+limit < len(records) else None)


def read_raw_reference(context, reference_id, include_details=False):
    record = context.get('reference_index', {}).get(reference_id)
    if record is None:
        return dict(status='needs_input', blocking=False, hint='Use a reference_id returned by this bound raw comparison')
    from molsteer.agents.design_audit import bounded_values
    return dict(status='observed_reference', comparison_id=context.get('comparison_id'), reference_id=reference_id,
                evidence=deepcopy(record) if include_details else bounded_values(record),
                limit='This cross-time reference does not authorize current numerical reward binding.')
