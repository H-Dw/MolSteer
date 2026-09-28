"""Constrained synthesis: facts -> retrieval -> applicability -> executable contract."""
import math
from molreader.localized_report import validate_localized_report
from molsteer.common import digest
from .knowledge import KnowledgeBase
from molsteer.contracts import validate_enriched, validate_reward_schema


def _requirements(entry):
    reasons = {
        'G03': 'No anchor coordinates or mobility declaration', 'G04': 'No target directions or prepared directional features',
        'G05': 'No indicated stereo/plane defect with validated reference', 'G06': 'No selected reference shape or alignment',
        'G07': 'No differentiable environment descriptor or reference library',
        'P01': 'Global strain is evidence, but stable graph/protonation and parameter applicability are unverified',
        'P02': 'Protein coordinates exist; receptor force-field types do not',
        'P03': 'Formal charges cannot replace atomistic partial charges or a dielectric model',
        'P04': 'No xTB oracle, verified chemistry or oracle budget',
        'P05': 'No prepared Vina inputs/backend or live suffix derivative',
        'P06': 'No atom-group SASA target or differentiable backend',
        'P07': 'No aligned ESP reference or partial charges',
        'S01': 'No target value/tolerance or live population',
        'S02': 'Affinity numbers alone do not define objectives, temperature or live particle population',
        'S03': 'No same-candidate matched off-target scores',
        'S04': 'Estimator, not reward; missing reevaluable oracle, perturbable runtime state and budget',
        'S05': 'No declared multiobjective population or diversity metric',
        'S06': 'Saved individual snapshots do not provide calibrated editing-depth batches',
        'S07': 'No target fragment and no three-population search adapter',
    }
    return reasons.get(entry['function_id'], 'No supported localized evidence for this term')


def derive(packet, report, knowledge_path):
    validate_enriched(packet)
    validate_localized_report(report, packet)
    if 'steering' not in packet:
        raise ValueError('MolReader steering enrichment is required')
    kb = KnowledgeBase(knowledge_path)
    categories = [c['category'] for c in report['findings'] + report['raw_state_findings']]
    # Metric tags connect localized report categories to the reviewed knowledge rows.
    if any(x['metric_id'] == 'protein_clashes' for x in report['evidence_index'].values()):
        categories.append('protein_clashes')
    retrieved = kb.retrieve(categories)
    entries = {x['function_id']: x for x in retrieved}
    terms, deferred = [], []
    idx = report['evidence_index']
    for card in report['findings'] + report['raw_state_findings']:
        view = 'state' if card['scope'] == 'observed_noisy_state' else 'prediction'
        candidates = [idx[eid] for eid in card['primary_evidence_ids'] if idx[eid]['view'] == view]
        angle_roles = {tuple(x['evidence']['atom_ids']) for x in candidates if x['metric_id'] == 'bond_angles'}
        bond_roles = {tuple(x['evidence']['atom_ids']) for x in candidates if x['metric_id'] == 'bond_lengths'}
        for x in candidates:
            e, m = x['evidence'], x['metric_id']
            term = None
            if view == 'prediction' and m == 'bond_angles' and e.get('endpoint_lower_angstrom') is not None:
                term = dict(family='flat_bottom_distance', atom_ids=[e['atom_ids'][0], e['atom_ids'][2]],
                    lower=e['endpoint_lower_angstrom'], upper=e['endpoint_upper_angstrom'], scale=1., unit='angstrom',
                    observable='1-3 endpoint distance; coupled bond-length/angle proxy', graph_dependent=True)
            elif view == 'prediction' and m == 'bond_lengths':
                term = dict(family='flat_bottom_distance', atom_ids=e['atom_ids'], lower=e['lower_angstrom'],
                    upper=e['upper_angstrom'], scale=1., unit='angstrom', observable='bond length', graph_dependent=True)
            elif view == 'prediction' and m == 'mmff_local_geometry':
                if e['kind'] == 'bond_angle' and tuple(e['atom_ids']) in angle_roles:
                    deferred.append(dict(evidence_id=e['evidence_id'], reason='Same angle already represented by its 1-3 distance; retained as support, no additional weight'))
                    continue
                if e['kind'] == 'bond_length' and tuple(e['atom_ids']) not in bond_roles:
                    ref, tol = e['reference_angstrom'], x['thresholds']['bond_relative_deviation']
                    term = dict(family='flat_bottom_distance', atom_ids=e['atom_ids'], lower=ref*(1-tol), upper=ref*(1+tol),
                        scale=1., unit='angstrom', observable='MMFF reference bond window', graph_dependent=True)
                elif e['kind'] == 'bond_angle':
                    ref, tol = e['reference_degrees'], x['thresholds']['angle_absolute_deviation_degrees']
                    term = dict(family='flat_bottom_angle', atom_ids=e['atom_ids'], lower=max(0.,ref-tol), upper=min(180.,ref+tol),
                        scale=10., unit='degree', observable='MMFF reference angle window', graph_dependent=True)
            elif m == 'protein_clashes' and packet['representations'][view]['coordinate_frame'] == 'receptor_world':
                atom = next((a for a in packet['steering']['receptor_atoms'] if a['serial'] == e['receptor_serial'] and a['residue_id'] == e['residue_id']), None)
                if atom:
                    term = dict(family='minimum_distance', atom_ids=e['atom_ids'], reference_coords=atom['coords'],
                        receptor_identity={k:atom[k] for k in ['serial','residue_id','atom_name']},
                        lower=e['distance_angstrom']+e['penetration_angstrom'], upper=None, scale=1., unit='angstrom',
                        observable='ligand-protein minimum distance', graph_dependent=False)
            if term:
                fid = 'G02' if term['family'] == 'minimum_distance' else 'G01'
                source_evidence = [eid for eid in card['evidence_ids'] if idx[eid]['metric_id'] == m and idx[eid]['evidence']['atom_ids'] == e['atom_ids']]
                term.update(term_id='term_'+digest(dict(view=view, metric=m, evidence=e['evidence_id']))[:12],
                    view=view, function_id=fid, finding_id=card['finding_id'], evidence_ids=source_evidence,
                    reference_evidence_id=e['evidence_id'], hypothesis_atom_ids=e['atom_ids'], weight=1.,
                    parameter_origin='Reported screening bounds; normalization/weight are explicit uncalibrated demonstration settings',
                    knowledge_source=entries[fid]['source'],
                    mathematical_form='E = 0.5*w*(relu((lower-v)/scale)^2 + relu((v-upper)/scale)^2); R = -E',
                    derivation='Squared-hinge specialization of interval deviation, zero throughout the accepted window; upper term omitted for minimum distance',
                    differentiability='Piecewise smooth in coordinates; angle requires nondegenerate vectors; category argmax is frozen',
                    local_chemical_context=card['chemical_context'],
                    conditions=['Current element/charge/graph hypothesis remains unchanged', 'Bounds are screening references, not validated physical acceptance limits'] if term['graph_dependent'] else ['Current element radii and aligned pocket remain applicable'],
                    executable_scope='offline_coordinate_copy_only')
                terms.append(term)
    # Reward groups never add raw-state energy to predicted-endpoint energy.
    groups = [dict(view=v, term_ids=[t['term_id'] for t in terms if t['view']==v],
                   reward='negative_sum_of_dimensionless_penalties') for v in ['prediction','state'] if any(t['view']==v for t in terms)]
    selected = {t['function_id'] for t in terms}
    for r in retrieved:
        r['decision'] = 'conditional_offline' if r['function_id'] in selected else 'deferred'
        r['decision_reason'] = 'Localized measurements and bounds available; frozen hypothesis and explicit demo mobility required' if r['function_id'] in selected else _requirements(r)
        r['backend_implemented'] = r['function_id'] in {'G01','G02'}
    unhandled = [dict(finding_id=c['finding_id'], category=c['category'], scope=c['scope'],
                     reason='Retained diagnosis; no supported coordinate reward synthesized for this finding')
                 for c in report['findings'] + report['raw_state_findings'] if not any(t['finding_id']==c['finding_id'] for t in terms)]
    spec = dict(kind='RewardSpec', schema_version='1.0.0', packet_id=packet['packet_id'], identity=packet['identity'],
        knowledge_source=dict(file=kb.path.name,sha256=kb.sha256,entry_count=len(kb.entries)),
        generation=packet['steering']['generation'], graph_signatures=packet['steering']['graph_signatures'],
        coordinate_hashes={v:s['coordinate_hash'] for v,s in packet['steering']['coordinate_snapshots'].items()},
        goal='Reduce diagnosed local geometry penalties conditionally on current chemical hypotheses',
        terms=terms, reward_groups=groups, retrieval= retrieved, deferred_evidence=deferred, unhandled_findings=unhandled,
        runtime_execution=dict(status='blocked', blockers=['No live generator Jacobian/suffix adapter', 'No declared editable/fixed masks or control budget'],
            possible_mapping='For an attached live graph only: grad_Xt R = J_endpoint(Xt)^T grad_endpoint R; flow update needs generator-specific sign/scale adapter'),
        assumptions=['No property-target ranges invented', 'No affinity objective inferred', 'No softmax particle resampling inferred from three saved ligands',
            'Squared hinge is a derived interval-penalty variant, not a verbatim force-field energy', 'Weights/scales are not calibrated',
            'Raw-state and endpoint evaluations remain separate', 'Knowledge rows describe candidates, not instructions to execute'],
        required_validation=['Finite-difference coordinate gradient', 'Energy descent on a copy', 'Graph/reference validity after topology changes', 'Full sampler ablation remains outstanding'])
    spec['reward_id'] = 'rw_'+digest(spec)[:24]
    validate_spec(spec, packet, report)
    return spec


def validate_spec(spec, packet, report=None):
    validate_enriched(packet)
    validate_reward_schema(spec)
    if spec['packet_id'] != packet['packet_id'] or spec['identity'] != packet['identity']:
        raise ValueError('Reward/packet identity mismatch')
    evidence = {e['evidence_id'] for m in packet['observations'] for e in m['evidence']}
    ids = set()
    for term in spec['terms']:
        if term['term_id'] in ids:
            raise ValueError('Duplicate reward term')
        ids.add(term['term_id'])
        if not set(term['evidence_ids']) <= evidence or term['reference_evidence_id'] not in evidence:
            raise ValueError('Unbound reward evidence')
        if not set(term['hypothesis_atom_ids']) <= set(packet['representations'][term['view']]['original_atom_ids']):
            raise ValueError('Unknown reward atom slot')
        if term['scale'] <= 0 or term['weight'] < 0 or not all(math.isfinite(term[k]) for k in ['lower','scale','weight']):
            raise ValueError('Invalid reward parameter')
        if term['upper'] is not None and (not math.isfinite(term['upper']) or term['upper'] < term['lower']):
            raise ValueError('Invalid interval')
    group_ids=[term_id for g in spec['reward_groups'] for term_id in g['term_ids']]
    if set(group_ids)!=ids or len(group_ids)!=len(ids):
        raise ValueError('Each term must belong to exactly one reward group')
    for group in spec['reward_groups']:
        if any(t['view']!=group['view'] for t in spec['terms'] if t['term_id'] in group['term_ids']):
            raise ValueError('Cannot mix representations in one reward group')
    return True
