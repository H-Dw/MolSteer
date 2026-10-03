from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from molsteer.common import digest
from molsteer.molreader.residual_needs import build_residual_needs
from molsteer.molreader.raw_reference import inspect_comparison
from molsteer.agents.measurement_supplements import MeasurementSupplements


def context(states, final=True, retyped=False):
    times = [.17, .42, .68, .94][:len(states)]
    nodes = [dict(node_id=str(i), time=t, role='anchor' if i == 0 else
        'final' if final and i == len(states)-1 else 'intermediate') for i,t in enumerate(times)]
    observations = [dict(node_id=str(i), time=t, current_condition_status=state,
        status='chemical_identity_changed' if retyped and i else state,
        current_atom_types=[dict(atom_id=2, element='N' if retyped and i else 'C')],
        current_reference_parameters=[dict(reference_angstrom=1.4+i/10)],
        measurement_reference_ids=[str(i)], current_condition_reference_ids=[]) for i,(t,state) in enumerate(zip(times,states))]
    return dict(status='available', nodes=nodes, reference_index={str(i):dict(record={'value':3.,'reference':1.4}) for i in range(len(states))},
        risk_tracks=[dict(track_id='risk',metric_id='geometry',view='prediction',factors=['geometry'],atom_ids=[2],
                         observations=observations,current_evidence_ids=['current'],current_measurement_ids=['now'])], opportunities=[])


@pytest.mark.parametrize('states,expected', [
    (['flagged','flagged','flagged'], 'persistent'),
    (['flagged','not_flagged','flagged'], 'recurrent'),
    (['not_flagged','not_flagged','flagged'], 'late_emergent'),
    (['flagged','flagged','not_flagged'], 'naturally_resolved'),
    (['flagged','relation_absent'], 'relation_absent_at_final'),
    (['flagged','measurement_unavailable'], 'terminal_unobserved_or_unavailable')])
def test_terminal_first_classifications_do_not_disappear_after_retyping(states, expected):
    raw = context(states, retyped=True)
    card, = build_residual_needs(raw)
    assert card['evolution'] == expected
    assert card['chemistry_transitions']
    assert card['current_precursor']['measurement_ids'] == ['now']
    assert card['observations'][-1]['current_atom_types'][0]['element'] == 'N'
    assert card['observations'][-1]['measurements'][0]['value'] == 3.


def test_missing_final_is_not_last_visible_node_and_cards_are_paged():
    raw = context(['flagged','not_flagged'], final=False)
    raw['residual_needs'] = build_residual_needs(raw)
    assert raw['residual_needs'][0]['terminal_status'] == 'not_observed'
    assert raw['residual_needs'][0]['terminal_node_id'] is None
    raw['residual_needs'] *= 3
    result = inspect_comparison(raw, section='residual_needs', offset=1, limit=1)
    assert result['total_count'] == 3 and result['next_offset'] == 2


def test_residual_index_keeps_terminal_and_current_evidence_without_repeating_full_measurements():
    raw = context(['flagged','not_flagged','flagged'], retyped=True)
    raw['residual_needs'] = build_residual_needs(raw)
    before = deepcopy(raw)
    compact = inspect_comparison(raw, section='residual_needs')['records'][0]
    full = inspect_comparison(raw, section='residual_needs', include_details=True)['records'][0]
    assert compact['terminal_status'] == full['terminal_status'] == 'flagged'
    assert compact['evolution'] == full['evolution'] == 'recurrent'
    assert compact['current_precursor'] == full['current_precursor']
    assert compact['chemistry_transitions'] == full['chemistry_transitions']
    for brief, original in zip(compact['observations'], full['observations']):
        assert brief['measurement_reference_ids'] == original['measurement_reference_ids']
        assert brief['current_condition_status'] == original['current_condition_status']
        assert 'measurements' not in brief and original['measurements']
    assert raw == before


def test_existing_calculator_creates_separate_bound_supplement_and_cache(tmp_path):
    coords = np.array([[0.,0.,0.],[1.2,0.,0.]])
    identity = dict(target_id='target',ligand_id='ligand',stage='stage',stage_t=.27)
    packet = dict(packet_id='packet',identity=identity,steering=dict(coordinate_snapshots=dict(prediction=dict(
        atom_ids=[0,1],coords_angstrom=coords.tolist(),coordinate_hash=digest(coords.tolist())))))
    original = deepcopy(packet)
    ctx = SimpleNamespace(identity=identity, coords=coords, atom_ids=np.array([0,1]), atoms=['C','N'],
        config={},sources={'original':{'sha256':'source_digest'}},view='prediction',options={})
    state = {}
    service = MeasurementSupplements(packet, {}, state, tmp_path, tmp_path/'output', contexts={('anchor','prediction'):ctx})
    result = service.measure('anchor','atom_inventory','prediction',[1])
    assert result['status'] == 'ok' and not result['cached']
    second = service.measure('anchor','atom_inventory','prediction',[1])
    assert second['cached'] and second['supplement_id'] == result['supplement_id']
    stored = state['evidence_supplements'][result['supplement_id']]
    assert stored['measurement']['values']['elements'] == {'C':1,'N':1}
    assert stored['binding']['coordinate_hash'] == packet['steering']['coordinate_snapshots']['prediction']['coordinate_hash']
    assert packet == original and len(list((tmp_path/'output').glob('*.json'))) == 1
    ctx.coords = coords + 1
    assert service.measure('anchor','atom_inventory','prediction',[1])['status'] == 'unavailable'
    assert service.measure('anchor','invented_calculator','prediction',[])['status'] == 'unavailable'
    with pytest.raises(ValueError, match='outside data'):
        MeasurementSupplements(packet, {}, {}, tmp_path, tmp_path/'data'/'supplements')
