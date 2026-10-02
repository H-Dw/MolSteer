"""Reject incomplete physical predicates, invented tolerances and false shapes."""
import copy
import json
import pytest
from test_expert_system import case
from molsteer.agents.audit_contracts import FACTORS
from molsteer.agents.design_audit import expression_constants, biophysical_context, bounded_values, inspect_bound_measurements, measurement_catalog, replace_draft_fields
from molsteer.agents.expert_contracts import validate_biology, validate_math
from molsteer.agents.loop import _validation_feedback
from molsteer.molreader.measurement_refs import measurement_references


def audited(case):
    packet, report, biology, design, retrieval = copy.deepcopy(case)
    direction = design['directions'][0]
    evidence_ids = biology['directions'][0]['evidence_ids']
    # The fixture's selected direction is not necessarily its first alert.
    evidence_ids = next(d['evidence_ids'] for d in biology['directions'] if d['direction_id']=='repair')
    repair=next(d for d in biology['directions'] if d['direction_id']=='repair')
    repair['repair_clauses']=[dict(clause_id='distance_window',observable_kind='distance',
        evidence_ids=direction['observables'][0]['evidence_ids'],
        predicate='The declared local distance lies within its synthetic accepted interval.')]
    direction['predicate_coverage']=[dict(clause_id='distance_window',
        observable_ids=[direction['observables'][0]['observable_id']],
        implementation='The interval residual is exactly zero throughout the declared window.')]
    biology['factor_assessment'] = [dict(factor=factor,
        disposition='optimize' if factor=='bond_geometry' else 'monitor',
        direction_ids=['repair'] if factor=='bond_geometry' else [],
        evidence_ids=evidence_ids if factor=='bond_geometry' else [],
        assessment='A synthetic factor audit retains independent unavailable outcomes as monitoring.',
        missing_requirements=[], shortcut_test='Distance alone cannot certify a coupled global physical outcome.') for factor in FACTORS]
    design['design_audit'] = dict(execution_scope='bounded_hypothesis_pilot',
        graph_policy='suspend_on_graph_change', architectures=[dict(name='local interval',
            mathematical_form='Half squared interval-distance deficit.', decision='selected',
            reason='Synthetic shape tests target only the declared local distance.')],
        selection_reason='One synthetic target with explicitly uncalibrated screening bounds.',
        unsupported_claims=['No claim of global strain repair or improved binding.'])
    direction['source_transform']='specialized'
    direction['function_basis']=[dict(locator=direction['function_lineage'][0]['locator'],
        role='direct_repair',decision='selected',reason='Inspected interval shape is used in this synthetic fixture.',missing_requirements=[])]
    constants = expression_constants(direction['expression'])
    unique = {(c['origin'],c['unit'],c['value']):c for c in constants}
    direction['reference_parameters']=[dict(origin=c['origin'],unit=c['unit'],value=c['value'],
        role='tolerance',provenance='declared_assumption',evidence_ids=[],source_locators=[],
        derivation='Declared uncalibrated coordinate-copy test value, not a physical calibration.') for c in unique.values()]
    lower, upper = constants[1]['value'], constants[3]['value']
    ident=direction['observables'][0]['observable_id']
    direction['shape_probes']=[dict(observable_values={ident:value},expected_zero=zero,
        derivative_signs={ident:sign},reason='Probe the repaired interval and each violated side.')
        for value,zero,sign in [(lower-.1,False,'negative'),((lower+upper)/2,True,'zero'),(upper+.1,False,'positive')]]
    return packet, report, biology, design, retrieval


def check(case):
    p,r,b,d,ret=case
    b=validate_biology(b,r,p,require_audit=True)
    return validate_math(d,b,p,[ret],{},report=r,require_audit=True)


def test_audited_function_has_actual_zero_set_and_derivative_signs(case):
    result=check(audited(case))
    assert result['design_audit']['execution_scope']=='bounded_hypothesis_pilot'


def test_clause_allows_additional_observable_evidence_without_losing_its_binding(case):
    value=audited(case)
    obs=value[3]['directions'][0]['observables'][0]
    extra=next(ident for ident,ref in measurement_references(value[0]).items()
               if ref['view']==obs['view'] and not set(ref['atom_ids']) & set(obs['atom_ids']))
    repair=next(d for d in value[2]['directions'] if d['direction_id']=='repair')
    repair['evidence_ids'].append(extra)
    obs['evidence_ids'].append(extra)
    assert check(value)['directions'][0]['status']=='executable'
    # An overlapping citation alone is insufficient when it does not localize
    # the clause's actual atoms, even though other observable evidence does.
    repair['repair_clauses'][0]['evidence_ids']=[extra]
    with pytest.raises(ValueError,match='no implicit host gates'):
        check(value)


def test_missing_factors_cannot_hide_unconsidered_biology(case):
    value=audited(case);value[2]['factor_assessment'].pop()
    with pytest.raises(ValueError,match='every biophysical factor'):check(value)


def test_missing_repair_conjunct_cannot_hide_an_unimplemented_condition(case):
    value=audited(case)
    repair=next(d for d in value[2]['directions'] if d['direction_id']=='repair')
    repair['repair_clauses'].append(dict(clause_id='global_area',observable_kind='burial_sasa',
        evidence_ids=repair['evidence_ids'],predicate='Retain the complete measured interface buried area above its target.'))
    with pytest.raises(ValueError,match='every declared repair clause'):check(value)


def test_unsupported_area_clause_cannot_be_claimed_as_an_implicit_host_gate(case):
    value=audited(case)
    repair=next(d for d in value[2]['directions'] if d['direction_id']=='repair')
    repair['repair_clauses'][0]['observable_kind']='burial_sasa'
    value[3]['directions'][0]['predicate_coverage'][0]['implementation']='An imagined host gate verifies all surface conditions outside the expression.'
    with pytest.raises(ValueError,match='no implicit host gates'):check(value)


def test_active_biology_must_enumerate_its_repair_predicate(case):
    value=audited(case)
    next(d for d in value[2]['directions'] if d['direction_id']=='repair')['repair_clauses']=[]
    with pytest.raises(ValueError,match='independent repair clause'):check(value)


def test_measurement_batch_preserves_support_bounds_summaries_and_never_mutates_packet(case):
    from molsteer.common import digest
    packet=case[0];before=digest(packet);refs=measurement_references(packet)
    summary=next(r for r in refs.values() if r['metric_id']=='protein_contacts' and not r['path'])
    row=next(r for r in refs.values() if r['metric_id']=='protein_contacts' and r['path'])
    result=inspect_bound_measurements(refs,[summary['evidence_id'],row['evidence_id']])
    assert result['requested_count']==2
    assert result['measurements'][0]['values']['contacts']['truncated']
    assert result['measurements'][1]==row
    full=inspect_bound_measurements(refs,[summary['evidence_id']],True)
    assert full['measurements'][0]==summary and digest(packet)==before
    assert bounded_values([1,2,3])==[1,2,3]
    with pytest.raises(ValueError) as error:inspect_bound_measurements(refs,[row['evidence_id'],'unknown'])
    assert _validation_feedback(error.value)['validation_path']==['evidence_ids',1]


def test_unknown_biology_evidence_has_exact_field_path_without_echoing_value(case):
    packet,report,biology,*_=copy.deepcopy(case)
    biology['directions'][0]['evidence_ids'].append('PRIVATE_UNKNOWN_VALUE')
    with pytest.raises(ValueError) as error:validate_biology(biology,report,packet)
    feedback=_validation_feedback(error.value)
    assert feedback['validation_path']==['directions',0,'evidence_ids',len(biology['directions'][0]['evidence_ids'])-1]
    assert 'PRIVATE_UNKNOWN_VALUE' not in json.dumps(feedback)


def test_draft_replacements_are_atomic_json_only_and_do_not_mutate_source():
    draft={'directions':[{'expression':{'op':'constant','value':1}}]}
    replacement={'path':['directions',0,'expression','value'],'value':2}
    assert replace_draft_fields(draft,[replacement])['directions'][0]['expression']['value']==2
    assert draft['directions'][0]['expression']['value']==1
    with pytest.raises(ValueError):replace_draft_fields(draft,[replacement,{'path':['directions',-1],'value':{}}])
    assert draft['directions'][0]['expression']['value']==1


def test_direction_staging_preserves_model_authorship_and_requires_complete_validation(case):
    from molsteer.agents.design_audit import stage_direction_draft, assemble_direction_drafts
    p,r,b,d,ret=audited(case);staged={}
    direction=copy.deepcopy(d['directions'][0]);metadata={k:v for k,v in d.items() if k not in ('kind','schema_version','directions')}
    with pytest.raises(ValueError,match='each selected direction'):assemble_direction_drafts(staged,b,metadata)
    result=stage_direction_draft(staged,direction,{direction['direction_id']})
    assert result['status']=='draft_only' and result['validated'] is False
    direction['expression']=None
    assembled=assemble_direction_drafts(staged,b,metadata)
    assert assembled['directions']==d['directions'] and all(assembled[k]==v for k,v in metadata.items())
    assert check((p,r,b,assembled,ret))
    # Staging retains an invalid draft for repair, but the unchanged full check rejects it.
    stage_direction_draft(staged,direction,{direction['direction_id']})
    with pytest.raises(ValueError,match='all prerequisites'):check((p,r,b,assemble_direction_drafts(staged,b,metadata),ret))
    with pytest.raises(ValueError,match='selected by'):stage_direction_draft(staged,{'direction_id':'unknown'},{direction['direction_id']})


def test_global_clash_guarantee_cannot_be_hidden_outside_repair_clauses(case):
    value=audited(case)
    repair=next(d for d in value[2]['directions'] if d['direction_id']=='repair')
    repair['repair_predicate']+=' AND no new protein clashes.'
    with pytest.raises(ValueError,match='Global repair guarantees'):check(value)


def test_math_cannot_reintroduce_unimplemented_monitor_gates_after_biology_review(case):
    value=audited(case)
    value[3]['directions'][0]['acceptable_set']+='; monitors (strain proxy, chirality and buried fraction) must show no regression.'
    with pytest.raises(ValueError,match='Independent monitors'):check(value)
    value[3]['directions'][0]['acceptable_set']='Only the declared local distance interval is enforced; independent monitor evaluation is not_run.'
    assert check(value)


def test_typed_expression_rejects_unknown_ops_and_preserves_legacy_numeric_content(case):
    from molsteer.agents.expert_contracts import MathematicalDesign
    design=copy.deepcopy(case[3])
    assert MathematicalDesign.model_validate(design).model_dump()['directions'][0]['expression']==design['directions'][0]['expression']
    design['directions'][0]['expression']['op']='square'
    with pytest.raises(ValueError) as error:MathematicalDesign.model_validate(design)
    feedback=_validation_feedback(error.value)
    assert any(e['rule']=='union_tag_invalid' and e['path'][-1]=='expression' for e in feedback['validation_errors'])


def test_model_tool_schema_has_recursive_operator_arity_and_constant_units():
    from molsteer.agents.expert_contracts import MathematicalDesign
    schema=MathematicalDesign.model_json_schema();defs=schema['$defs']
    binary=defs['BinaryExpression']['properties']['args']
    assert binary['minItems']==binary['maxItems']==2
    assert defs['ConstantExpression']['properties']['unit']['enum']==['dimensionless','angstrom','angstrom^3','radian','degree','kcal/mol']
    assert 'maximum' in defs['BinaryExpression']['properties']['op']['enum']


def test_recursive_schema_error_path_is_a_real_json_path_for_draft_repair(case):
    from molsteer.agents.expert_contracts import MathematicalDesign
    design=copy.deepcopy(case[3]);node={'op':'constant','value':1,'unit':'dimensionless','origin':'test origin'}
    design['directions'][0]['expression']={'op':'maximum','args':[node,node,node]}
    with pytest.raises(ValueError) as error:MathematicalDesign.model_validate(design)
    assert _validation_feedback(error.value)['validation_errors']==[
        {'path':['directions',0,'expression','args'],'rule':'too_long'}]


def test_nonalert_measurement_can_support_a_preservation_direction(case):
    packet,report,biology,*_=copy.deepcopy(case)
    contact=next(ident for ident,ref in measurement_references(packet).items()
        if ref['metric_id']=='protein_contacts' and ref['view']=='prediction' and ref['path'])
    direction=copy.deepcopy(biology['directions'][0])
    direction.update(direction_id='preserve_contact',rank=len(biology['directions'])+1,
        finding_ids=[],evidence_ids=[contact],required=False,disposition='monitor')
    biology['directions'].append(direction)
    assert validate_biology(biology,report,packet)['directions'][-1]['evidence_ids']==[contact]
    # A finding-linked repair may also cite measured supporting context.
    biology['directions'][0]['evidence_ids'].append(contact)
    assert contact in validate_biology(biology,report,packet)['directions'][0]['evidence_ids']
    direction['evidence_ids']=['fabricated_nonalert_measurement']
    with pytest.raises(ValueError,match='unknown diagnostic evidence'):
        validate_biology(biology,report,packet)


def test_normal_contact_reference_binds_live_receptor_distance_without_mutation(case):
    from molsteer.common import digest
    from molsteer.molthinker.expressions import validate_observables
    packet=case[0];before=digest(packet)
    ref=next(ref for ref in measurement_references(packet).values()
        if ref['metric_id']=='protein_contacts' and ref['view']=='prediction' and ref['path'])
    observable=dict(observable_id='contact',kind='receptor_distance',view='prediction',
        atom_ids=ref['atom_ids'],evidence_ids=[ref['evidence_id']],
        parameters={key:ref[key] for key in ('receptor_serial','residue_id')})
    assert validate_observables([observable],packet,{ref['evidence_id']})=={'contact':'angstrom'}
    assert digest(packet)==before
    observable['atom_ids']=[next(i for i in packet['representations']['prediction']['original_atom_ids']
        if i not in ref['atom_ids'])]
    with pytest.raises(ValueError,match='not localized'):
        validate_observables([observable],packet,{ref['evidence_id']})


def test_derived_stereo_reference_matches_actual_tensor_order_and_coordinates(case):
    import torch
    from molsteer.common import digest
    from molsteer.molthinker.expressions import validate_observables,observable_value
    packet=case[0];before=digest(packet)
    ref=next(r for r in measurement_references(packet).values()
        if r['view']=='prediction' and r['kind']=='derived_geometry')
    obs=dict(observable_id='volume',kind='signed_volume',view='prediction',
        atom_ids=ref['atom_ids'],evidence_ids=[ref['evidence_id']],parameters={})
    assert validate_observables([obs],packet,{ref['evidence_id']})=={'volume':'angstrom^3'}
    snapshot=packet['steering']['coordinate_snapshots']['prediction']
    coords=torch.tensor(snapshot['coords_angstrom'],dtype=torch.float64,requires_grad=True)
    value=observable_value(obs,coords,snapshot['atom_ids'],packet)
    assert float(value.detach())==pytest.approx(ref['values']['signed_volume_angstrom3'])
    assert torch.isfinite(torch.autograd.grad(value,coords)[0]).all()
    assert digest(packet)==before and ref['coordinate_hash']==snapshot['coordinate_hash']


def test_distance_cannot_certify_global_strain_repair(case):
    value=audited(case)
    factor=next(f for f in value[2]['factor_assessment'] if f['factor']=='intramolecular_stability')
    factor.update(disposition='optimize',direction_ids=['repair'],evidence_ids=value[2]['factor_assessment'][0]['evidence_ids'])
    with pytest.raises(ValueError,match='physical repair predicate'):check(value)


def test_screening_band_is_not_a_calibrated_repair(case):
    value=audited(case);value[3]['design_audit']['execution_scope']='calibrated_repair'
    with pytest.raises(ValueError,match='bounded hypothesis pilot'):check(value)


def test_fake_observed_tolerance_is_rejected(case):
    value=audited(case)
    parameter=value[3]['directions'][0]['reference_parameters'][0]
    parameter.update(provenance='observed_reference',evidence_ids=[value[3]['directions'][0]['observables'][0]['evidence_ids'][0]])
    with pytest.raises(ValueError,match='occur in their cited evidence'):check(value)


def test_chemical_reference_requires_an_applicability_guard(case):
    value=audited(case)
    value[3]['design_audit']['graph_policy']='graph_independent'
    parameter=value[3]['directions'][0]['reference_parameters'][0]
    packet=value[0]
    chemical_evidence=next(e['evidence_id'] for m in packet['observations']
        if m['metric_id']=='bond_lengths' for e in m.get('evidence', []))
    parameter['evidence_ids']=[chemical_evidence]
    with pytest.raises(ValueError,match='graph-change applicability guard'):check(value)


def test_shape_summary_must_match_the_executable_derivative(case):
    value=audited(case)
    first=value[3]['directions'][0]['shape_probes'][0]
    first['derivative_signs']={k:'positive' for k in first['derivative_signs']}
    with pytest.raises(ValueError,match='derivative contradicts'):check(value)


def test_unknown_constant_provenance_is_actionable_and_redacted(case):
    value=audited(case);value[3]['directions'][0]['reference_parameters']=[]
    with pytest.raises(ValueError) as error:check(value)
    feedback=_validation_feedback(error.value)
    assert feedback['validation_path']==['directions',0,'reference_parameters']
    assert 'observations' not in json.dumps(feedback)


def test_context_includes_contacts_burial_and_unavailable_preparation(case):
    context=biophysical_context(case[0])
    assert {f['factor'] for f in context['factors']}==set(FACTORS)
    contact=next(f for f in context['factors'] if f['factor']=='target_contacts')
    assert any(o['metric_id']=='protein_contacts' for o in contact['observations'])
    assert 'coordinate derivatives only' in context['categorical_control']


def test_context_bounds_nested_evidence_and_does_not_duplicate_summary_values(case):
    from molsteer.common import digest
    packet=copy.deepcopy(case[0])
    metric=next(m for m in packet['observations'] if m['view']=='prediction' and m['metric_id']=='protein_contacts')
    metric['evidence']=[{'evidence_id':'synthetic-context-evidence','atom_ids':[],
                        'details':{'many_rows':[{'distance':i} for i in range(1000)]}}]
    before=digest(packet);context=biophysical_context(packet,'target_contacts')
    observation=next(o for o in context['factors'][0]['observations'] if o['metric_id']=='protein_contacts' and o['view']=='prediction')
    assert observation['evidence'][0]['details']['many_rows']['total_count']==1000
    assert len(observation['evidence'][0]['details']['many_rows']['items'])==4
    assert all('values' not in r for r in observation['measurement_references']['references'])
    assert digest(packet)==before


def test_paged_measurement_catalog_keeps_every_exact_reference_retrievable(case):
    refs=measurement_references(case[0]);seen=[];offset=0
    while True:
        page=measurement_catalog(refs,'protein_contacts',limit=3,offset=offset)
        seen.extend(r['evidence_id'] for r in page['references'])
        if page['next_offset'] is None:break
        offset=page['next_offset']
    assert set(seen)=={ident for ident,r in refs.items() if r['metric_id']=='protein_contacts' and r['view']=='prediction'}
    localized=next(refs[ident] for ident in seen if refs[ident]['atom_ids'])
    filtered=measurement_catalog(refs,'protein_contacts',atom_id=localized['atom_ids'][0],limit=32)
    assert all(localized['atom_ids'][0] in r['atom_ids'] for r in filtered['references'])
    assert inspect_bound_measurements(refs,[localized['evidence_id']])['measurements'][0]==localized
    for arguments in ({'offset':-1},{'limit':33},{'view':'invented'}):
        with pytest.raises(ValueError):measurement_catalog(refs,'protein_contacts',**arguments)


def test_mmff_decomposition_cannot_be_claimed_as_a_distance_function(case):
    from molsteer.molthinker.research.corpus import MarkdownCorpus
    from test_expert_system import ROOT
    value=audited(case)
    source=MarkdownCorpus(ROOT/'knowledge').search('MMFF energy',function_id='P01')['records'][0]
    value[4]['records'].append(source)
    direction=value[3]['directions'][0]
    direction['function_lineage']=[dict(source_id=source['source_id'],locator=source['chunk_id'],
        original_formula=source['formula'],adaptation='Claim the full energy is represented by one distance; this must fail.')]
    direction['source_ids']=[source['source_id']]
    direction['function_basis'][0]['locator']=source['chunk_id']
    with pytest.raises(ValueError,match='complete MMFF energy'):check(value)
