"""Saved raw comparisons and actual tool-loop wiring, without inference or paid APIs."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_expert_system import case, make_models

from molsteer.common import digest
from molsteer.contracts import validate_enriched
from molsteer.agents.config import load_config, RawReferenceConfig
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.reader import reader_tools
from molsteer.agents.design_audit import _evidence
from molsteer.agents.decision_workspace import current_state_context, new_workspace, update_workspace
from molsteer.agents.expert_contracts import ModelDynamicsContext, validate_biology
from molsteer.molreader.raw_reference import (load_raw_reference,
    inspect_comparison, read_raw_reference, bound_raw_context)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/5i0b_A__5vef_M77/ligand_002/t_0.50/StatePacket.json'


def metric(packet, name, view='prediction'):
    return next(m for m in packet['observations'] if m['metric_id'] == name and m['view'] == view)


def stamp(packet, time=None):
    if time is not None:
        packet['identity'].update(stage=f'synthetic_{time}', stage_t=time)
        packet['steering']['generation']['t']['value'] = time
    for m in packet['observations']:
        m['identity'] = copy.deepcopy(packet['identity'])
    packet['coverage'] = {s: sum(m['status'] == s for m in packet['observations']) for s in packet['coverage']}
    # Distinguish synthetic fixture provenance; no real source file is modified.
    packet['parent_packet_id'] = 'sp_' + digest(packet['observations'])[:24]
    packet['packet_id'] = 'sp_' + digest({'parent': packet['parent_packet_id'], 'steering': packet['steering']})[:24]
    validate_enriched(packet)
    return packet


def save(root, name, value):
    (root / name).write_text(json.dumps(value), encoding='utf-8')
    return name


def setup_raw(tmp_path, anchor=None, times=(.23, .61, .93), direction='increasing'):
    original = json.loads(EXAMPLE.read_text(encoding='utf-8'))
    anchor = stamp(copy.deepcopy(original), times[0]) if anchor is None else copy.deepcopy(anchor)
    middle, final = [stamp(copy.deepcopy(anchor), t) for t in times[1:]]
    # Native repair only first visible at the final sample; another angle risk persists.
    local = metric(final, 'mmff_local_geometry')
    local['evidence'] = [e for e in local['evidence'] if e.get('kind') != 'bond_length']
    for row in local['values']['bonds']:
        if row['atom_ids'] == [1, 14]:
            row.update(distance_angstrom=row['reference_angstrom'], relative_deviation=0.0)
    for index, packet in enumerate((middle, final), 1):
        metric(packet, 'affinity')['values']['pkd'] += index * .3
        metric(packet, 'sa_score')['values']['sa_score'] -= index * .2
        metric(packet, 'mmff_strain')['values']['strain_proxy_kcal_mol'] -= index * 5
        contacts = metric(packet, 'protein_contacts')['values']['contacts']
        new = copy.deepcopy(contacts[0])
        existing = {(r['atom_id'], r['receptor_serial']) for r in contacts}
        new['atom_id'] = next(a for a in range(20) if (a, new['receptor_serial']) not in existing)
        contacts.append(new)
        stamp(packet)
    manifest = dict(kind='RawInferenceManifest', schema_version='1.0', trajectory_id='synthetic_raw', mode='raw',
        time_direction=direction, anchor_packet_path=save(tmp_path, 'anchor.json', anchor),
        nodes=[dict(node_id='middle', time=times[1], role='intermediate', packet_path=save(tmp_path, 'middle.json', middle)),
               dict(node_id='end', time=times[2], role='final', packet_path=save(tmp_path, 'end.json', final))],
        provenance={'scope': 'synthetic stored-metric fixture, not a generated continuation'})
    save(tmp_path, 'manifest.json', manifest)
    config = RawReferenceConfig.model_validate(dict(enabled=True, manifest_path='manifest.json',
        reference_times=[times[1]], views=['prediction'], outcome_metrics=[
            dict(metric_id='affinity', value_path=['pkd'], direction='higher'),
            dict(metric_id='sa_score', value_path=['sa_score'], direction='lower'),
            dict(metric_id='mmff_strain', value_path=['strain_proxy_kcal_mol'], direction='lower')]))
    return anchor, middle, final, manifest, config


def track(context, kind):
    return next(r for r in context['risk_tracks'] if r['metric_id'] == 'mmff_local_geometry'
                and context['reference_index'][r['origin_reference_id']]['record'].get('kind') == kind)


def test_configurable_nodes_persistent_and_first_final_clearance(tmp_path):
    packet, _, _, _, cfg = setup_raw(tmp_path)
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    before = copy.deepcopy(packet)
    comparison = load_raw_reference(packet, cfg, tmp_path)
    assert comparison['status'] == 'available'
    assert [n['time'] for n in comparison['nodes']] == [.23, .61, .93]
    assert track(comparison, 'bond_length')['classification'] == 'first_clearance_at_raw_final'
    assert track(comparison, 'bond_length')['first_observed_clearance']['interval'] == [.61, .93]
    assert track(comparison, 'bond_angle')['classification'] == 'persistent_through_raw_final'
    assert {'additional_repair_assessment', 'earlier_repair_assessment', 'regional_enhancement_assessment'} <= {
        r['category'] for r in comparison['opportunities']}
    assert packet == before
    assert originals == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_selection_can_exclude_final_and_support_decreasing_time(tmp_path):
    packet, _, _, _, cfg = setup_raw(tmp_path, times=(.92, .43, .08), direction='decreasing')
    cfg.include_final = False
    comparison = load_raw_reference(packet, cfg, tmp_path)
    assert comparison['status'] == 'available'
    assert [n['time'] for n in comparison['nodes']] == [.92, .43]
    assert all('raw_final' not in r['classification'] for r in comparison['risk_tracks'])
    cfg.include_final = True
    assert track(load_raw_reference(packet, cfg, tmp_path), 'bond_length')['first_observed_clearance']['interval'] == [.43, .08]


@pytest.mark.parametrize('change', ['unavailable', 'partial_missing_row', 'method_changed', 'threshold_changed', 'local_reference_changed', 'retyped'])
def test_missing_or_incomparable_is_not_repair(tmp_path, change):
    packet, _, final, _, cfg = setup_raw(tmp_path)
    local = metric(final, 'mmff_local_geometry')
    if change == 'unavailable':
        local.update(status='unavailable', values={}, evidence=[])
    elif change == 'partial_missing_row':
        local['status'] = 'partial'
        local['values']['bonds'] = [r for r in local['values']['bonds'] if r['atom_ids'] != [1, 14]]
    elif change == 'method_changed':
        local['method'] = 'Different reference method'
    elif change == 'threshold_changed':
        local['thresholds']['bond_relative_deviation'] = .4
    elif change == 'local_reference_changed':
        row = next(r for r in local['values']['bonds'] if r['atom_ids'] == [1, 14])
        row['reference_angstrom'] += .2
    else:
        metric(final, 'chemistry_context')['values']['atoms'][1]['element'] = 'Cl'
    save(tmp_path, 'end.json', stamp(final))
    result = track(load_raw_reference(packet, cfg, tmp_path), 'bond_length')
    assert result['first_observed_clearance'] is None
    assert result['observations'][-1]['status'] not in ('flagged', 'not_flagged')
    assert result['classification'] not in ('first_clearance_at_raw_final', 'cleared_before_raw_final')


def test_missing_selected_files_are_kept_as_unknown_nodes(tmp_path):
    packet, _, _, manifest, cfg = setup_raw(tmp_path)
    manifest['nodes'][1]['packet_path'] = 'not_saved.json'
    save(tmp_path, 'manifest.json', manifest)
    result = load_raw_reference(packet, cfg, tmp_path)
    assert result['status'] == 'partial' and result['nodes'][-1]['status'] == 'unavailable'
    assert track(result, 'bond_angle')['classification'] != 'persistent_through_raw_final'
    cfg.reference_times = [.72]
    assert any(i['kind'] == 'requested_time_not_saved' for i in load_raw_reference(packet, cfg, tmp_path)['issues'])


@pytest.mark.parametrize('change', ['anchor_content', 'wrong_subject', 'wrong_time', 'wrong_mode', 'outside_path', 'bad_packet', 'bad_manifest'])
def test_provenance_and_saved_contracts_cannot_be_silently_mixed(tmp_path, change):
    packet, _, final, manifest, cfg = setup_raw(tmp_path)
    if change == 'anchor_content':
        packet['limitations'].append('Not the declared saved anchor')
    elif change == 'wrong_subject':
        final['identity']['ligand_id'] = 'different_sample'
        save(tmp_path, 'end.json', stamp(final))
    elif change == 'wrong_time':
        manifest['nodes'][1]['time'] = .97
    elif change == 'wrong_mode':
        manifest['mode'] = 'guided'
    elif change == 'outside_path':
        manifest['anchor_packet_path'] = '../outside.json'
    elif change == 'bad_packet':
        final['kind'] = 'OtherPacket'
        save(tmp_path, 'end.json', final)
    elif change == 'bad_manifest':
        manifest = []
    save(tmp_path, 'manifest.json', manifest)
    result = load_raw_reference(packet, cfg, tmp_path)
    assert result['status'] in ('unavailable', 'partial') and result['issues']
    assert not any(r['classification'] == 'persistent_through_raw_final' for r in result.get('risk_tracks', []))


def test_regional_association_is_not_benefit_and_refs_are_separate(tmp_path):
    packet, _, _, _, cfg = setup_raw(tmp_path)
    result = load_raw_reference(packet, cfg, tmp_path)
    outcomes = {r['metric_id']: r for r in result['outcome_trends']}
    assert all(r['observations'][-1]['directionally_improved_by_config'] for r in outcomes.values())
    new_contacts = [r for r in result['regional_changes'] if r['metric_id'] == 'protein_contacts'
                    and r['row_group'] == 'contacts' and not r['observations'][0]['listed']]
    assert new_contacts
    associations = [r for r in result['opportunities'] if r['category'] == 'regional_enhancement_assessment']
    assert associations and all(not r['auto_selected'] and r['intervention_effect'] == 'unobserved' for r in associations)
    assert all(r['evidence_class'] == 'raw_path_association' for r in associations)
    assert all(r['grouping'] and r['source_record_ids'] for r in associations)
    current_ids = set(_evidence(packet, {'evidence_index': {}}))
    assert set(result['reference_index']).isdisjoint(current_ids)
    assert set(result['factor_coverage']) == set(cfg.factors or result['factor_coverage'])
    preview = inspect_comparison(result, 'opportunities', limit=1)
    assert len(preview['records']) == 1 and preview['next_offset'] == 1
    future_id = next(i for i, r in result['reference_index'].items() if r['node_id'] == 'end')
    assert read_raw_reference(result, future_id)['evidence']['role'] == 'raw_reference'
    assert read_raw_reference(result, 'some_path')['blocking'] is False


def test_cross_view_associations_require_explicit_config_and_collation_keeps_sources(tmp_path):
    from molsteer.molreader.raw_reference import _group_regional_opportunities
    packet, middle, final, _, cfg = setup_raw(tmp_path)
    cfg.views = ['state', 'prediction']
    result = load_raw_reference(packet, cfg, tmp_path)
    regions = [r for r in result['opportunities'] if r['category'] == 'regional_enhancement_assessment']
    assert regions and all(r['view'] == 'prediction' for r in regions)
    assert all(a['representation_relation'] == 'same_view' for r in regions for a in r['associations'])
    cfg.cross_view_associations = True
    # An explicit state-only local change with a prediction-only measured score improvement.
    metric(final, 'protein_contacts', 'state')['values']['per_atom_nearest'][0]['nearest_distance_angstrom'] += .2
    save(tmp_path, 'end.json', stamp(final))
    result = load_raw_reference(packet, cfg, tmp_path)
    state_regions = [r for r in result['opportunities'] if r['category'] == 'regional_enhancement_assessment' and r['view'] == 'state']
    assert state_regions and any(a['representation_relation'] == 'cross_view_coevolution' for r in state_regions for a in r['associations'])
    # Grouping preserves distinct feature records and does not equate atom overlap with causality.
    sample = copy.deepcopy(regions[0])
    cfg.group_regional_candidates = False
    ungrouped = [r for r in load_raw_reference(packet, cfg, tmp_path)['opportunities'] if r['category'] == 'regional_enhancement_assessment']
    first = ungrouped[0]; second = copy.deepcopy(first)
    second.update(source_record_id='independent_feature', metric_id='other_metric')
    grouped = _group_regional_opportunities([first, second])
    assert len(grouped) == 1 and len(grouped[0]['source_record_ids']) == 2
    assert grouped[0]['metric_ids'] == [first['metric_id'], 'other_metric']


def test_analyses_factors_and_outcome_interpretation_are_configurable(tmp_path):
    packet, _, _, _, cfg = setup_raw(tmp_path)
    cfg.factors = ['bond_geometry']
    cfg.analyses = ['persistent_defects']
    cfg.outcome_metrics = []
    result = load_raw_reference(packet, cfg, tmp_path)
    assert not result['outcome_trends']
    assert all(r['factors'] == ['bond_geometry'] for r in result['regional_changes'])
    assert all(r['category'] == 'additional_repair_assessment' for r in result['opportunities'])
    assert not inspect_comparison(result, 'regional_changes', factor='target_contacts')['records']


def test_recurrence_and_sampling_gaps_remain_visible(tmp_path):
    packet, middle, final, _, cfg = setup_raw(tmp_path)
    # A gap between the last flag and first clear sample prevents precise repair timing.
    local = metric(middle, 'mmff_local_geometry')
    local.update(status='unavailable', evidence=[], values={})
    save(tmp_path, 'middle.json', stamp(middle))
    cleared = track(load_raw_reference(packet, cfg, tmp_path), 'bond_length')
    assert cleared['first_observed_clearance']['interval'] == [.23, .93]
    assert cleared['first_observed_clearance']['selected_measurement_gaps_between'] is True
    assert cleared['first_observed_clearance']['unsampled_steps'] == 'not_observed'
    # A previously cleared raw path that flags the defect again is not continuous persistence.
    metric(middle, 'mmff_local_geometry').update(copy.deepcopy(metric(final, 'mmff_local_geometry')))
    save(tmp_path, 'middle.json', stamp(middle))
    metric(final, 'mmff_local_geometry').update(copy.deepcopy(metric(packet, 'mmff_local_geometry')))
    save(tmp_path, 'end.json', stamp(final))
    recurred = track(load_raw_reference(packet, cfg, tmp_path), 'bond_length')
    assert recurred['classification'] == 'flagged_at_raw_final_with_gaps_or_recurrence'
    assert [r['status'] for r in recurred['observations']] == ['flagged', 'not_flagged', 'flagged']


def test_planarity_rows_have_localized_coverage_and_clearance(tmp_path):
    packet, middle, final, _, cfg = setup_raw(tmp_path)
    for index, p in enumerate((packet, middle, final)):
        planar = metric(p, 'ring_planarity')
        row = dict(atom_ids=[0, 7, 11] if index < 2 else [7, 11, 0], max_plane_distance_angstrom=.4 if index < 2 else .02,
                   rms_plane_distance_angstrom=.3 if index < 2 else .01)
        planar.update(status='ok', values={'rings': [row], 'aromatic_ring_count': 1, 'outlier_count': int(index < 2)},
                      evidence=[dict(**row, evidence_id='ev_' + digest(row)[:20], severity='warning', message='Nonplanar ring')] if index < 2 else [])
        stamp(p)
    save(tmp_path, 'anchor.json', packet); save(tmp_path, 'middle.json', middle); save(tmp_path, 'end.json', final)
    result = load_raw_reference(packet, cfg, tmp_path)
    planar_track = next(r for r in result['risk_tracks'] if r['metric_id'] == 'ring_planarity')
    assert planar_track['classification'] == 'first_clearance_at_raw_final'
    assert planar_track['observations'][-1]['measurement_reference_ids']


def test_target_context_and_absolute_energy_changes_are_not_comparable_outcomes(tmp_path):
    packet, _, final, _, cfg = setup_raw(tmp_path)
    # Same target label alone cannot verify the pocket geometry.
    final['steering']['receptor_atoms'][0]['coords'][0] += 1
    save(tmp_path, 'end.json', stamp(final))
    result = load_raw_reference(packet, cfg, tmp_path)
    affinity = next(r for r in result['outcome_trends'] if r['metric_id'] == 'affinity')['observations'][-1]
    assert affinity['comparison_status'] == 'receptor_context_unverified' and affinity['delta_from_anchor'] is None
    cfg.outcome_metrics = [type(cfg.outcome_metrics[0]).model_validate(dict(metric_id='mmff_strain',
        value_path=['energy_kcal_mol'], direction='lower'))]
    final['steering']['graph_signatures']['prediction'] = 'a' * 64
    save(tmp_path, 'end.json', stamp(final))
    result = load_raw_reference(packet, cfg, tmp_path)
    assert result['outcome_trends'][0]['observations'][-1]['comparison_status'] == 'absolute_energy_across_chemical_identities'


def test_disabled_and_stale_contexts_do_not_leak_future_facts(tmp_path):
    packet, _, final, _, cfg = setup_raw(tmp_path)
    context = load_raw_reference(packet, cfg, tmp_path)
    stale = bound_raw_context(context, final)
    assert stale['status'] == 'unavailable' and 'reference_index' not in stale
    current = current_state_context(final, {'findings': []}, ModelDynamicsContext().model_dump(), context)
    assert current['raw_reference']['counts']['opportunities'] == 0
    cfg.enabled = False
    result = load_raw_reference(packet, cfg, tmp_path)
    assert result['status'] == 'disabled' and 'reference_index' not in result


@pytest.mark.parametrize('fields', [dict(enabled=True), dict(reference_times=[.3, .3]), dict(views=[]),
    dict(factors=['imaginary']), dict(manifest_path='../anything.json'),
    dict(enabled=True, manifest_path='m.json', include_final=False)])
def test_invalid_configurations_are_rejected(fields):
    with pytest.raises(ValidationError):
        RawReferenceConfig.model_validate(fields)


def test_reader_current_diagnosis_stays_separate_and_new_calls_are_optional(case, tmp_path):
    packet, report = case[:2]
    _, _, _, _, cfg = setup_raw(tmp_path, packet, times=(packet['identity']['stage_t'], .68, .96))
    context = load_raw_reference(packet, cfg, tmp_path)
    tools, result = reader_tools(packet, report, raw_reference_context=context)
    catalog = {t.name: t for t in tools}
    raw = catalog['inspect_raw_comparison'].invoke({'section': 'risk_tracks'})
    assert raw['records']
    catalog['record_raw_reference_analysis'].invoke({'record': {'comparison_id': context['comparison_id'],
        'interpretation': 'Public sampled persistence interpretation.', 'private_reasoning': 'HIDDEN_THOUGHT'}})
    for name in ('inspect_geometry', 'inspect_chemistry', 'inspect_uncertainty', 'submit_diagnosis'):
        catalog[name].invoke({})
    assert result['report'] == report and result['raw_reference_context'] == context
    assert 'HIDDEN_THOUGHT' not in json.dumps(result['raw_reference_analysis'])
    # The comparison tools are helpful, never a new precondition to diagnosis submission.
    simple_tools, plain = reader_tools(packet, report, raw_reference_context=context)
    for t in simple_tools:
        if t.name in ('inspect_geometry', 'inspect_chemistry', 'inspect_uncertainty', 'submit_diagnosis'):
            t.invoke({})
    assert plain['report'] == report


def test_actual_reader_and_both_expert_loops_receive_temporal_context_and_public_review(case, tmp_path, monkeypatch):
    packet, report = case[:2]
    _, _, _, _, raw_cfg = setup_raw(tmp_path, packet, times=(packet['identity']['stage_t'], .67, .95))
    comparison = load_raw_reference(packet, raw_cfg, tmp_path)
    from molsteer.agents import runtime as runtime_module
    monkeypatch.setattr(runtime_module, 'load_raw_reference', lambda p, c, r: copy.deepcopy(comparison))
    cfg = load_config(); cfg.thinker.require_design_audit = False; cfg.thinker.external_research = False
    cfg.agents['molmonitor'].enabled = False; cfg.runtime.trace_dir = Path('outputs/test_raw_reference')
    models = make_models(case); seen = {}
    originals = {role: models[role].respond for role in ('molreader', 'molthinker.biology', 'molthinker.mathematics')}
    for role in originals:
        def capture(n, messages, role=role):
            if n == 1:
                seen[role] = (messages[0].content, json.loads(messages[1].content))
            if role == 'molreader' and n == 1:
                return [('inspect_raw_comparison', {'section': 'risk_tracks'}),
                        ('record_raw_reference_analysis', {'record': {'comparison_id': comparison['comparison_id'],
                            'temporal_observation': 'Some current risks persist along the sampled raw path.'}})]
            if role == 'molthinker.biology' and n == 1:
                return [('inspect_raw_comparison', {'section': 'opportunities'}),
                        ('record_biology_decision', {'stage': 'raw_reference_review', 'record': {
                            'comparison_id': comparison['comparison_id'],
                            'decision': 'Retain the current measured repair; raw association alone does not select another goal.'}})]
            return originals[role](n, messages)
        models[role].respond = capture
    state = AgentRuntime(cfg, models=models).run(packet, report, run_id='raw_reference_role_fixture')
    assert state['status'] == 'validated', state['trace'][-4:]
    for role in originals:
        assert seen[role][1]['raw_reference']['comparison_id'] == comparison['comparison_id']
        assert {'inspect_raw_comparison', 'read_raw_reference'} <= models[role].tools
    math_context = seen['molthinker.mathematics'][1]
    assert math_context['biology_decisions']['raw_reference_review']['comparison_id'] == comparison['comparison_id']
    assert math_context['raw_reference_analysis']['temporal_observation']
    assert state['diagnostic_report'] == report
    assert 'current observable evidence' in seen['molthinker.mathematics'][0]
    checkpoint = json.loads(Path(state['checkpoint_path']).read_text(encoding='utf-8'))
    assert checkpoint['artifacts']['raw_reference_context'] == comparison
    assert checkpoint['artifacts']['expert_workspace_history'][0]['raw_reference_binding']['comparison_id'] == comparison['comparison_id']
    assert all(not o['evidence_ids'] or not any(e.startswith('rr_') for e in o['evidence_ids'])
               for d in state['mathematical_design']['directions'] for o in d['observables'])


def test_temporal_ids_cannot_replace_current_biological_evidence(case, tmp_path):
    packet, report, biology = copy.deepcopy(case[:3])
    _, _, _, _, cfg = setup_raw(tmp_path, packet, times=(packet['identity']['stage_t'], .66, .94))
    context = load_raw_reference(packet, cfg, tmp_path)
    biology['directions'][0]['evidence_ids'] = [next(iter(context['reference_index']))]
    with pytest.raises(ValueError, match='unknown diagnostic evidence'):
        validate_biology(biology, report, packet)
    workspace = new_workspace()
    assert update_workspace(workspace, 'biology', 'raw_reference_review', {
        'comparison_id': context['comparison_id'], 'selected_goals': ['repair']})['blocking'] is False
