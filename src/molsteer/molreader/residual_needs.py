"""Terminal-first evidence cards; observations never assert intervention efficacy."""
from copy import deepcopy
from molsteer.common import digest


def build_residual_needs(context):
    nodes = context.get('nodes', [])
    terminal = next((n['node_id'] for n in nodes if n.get('role') == 'final'), None)
    references = context.get('reference_index', {})
    cards = []
    for track in context.get('risk_tracks', []):
        observations = deepcopy(track['observations'])
        states = [o['current_condition_status'] for o in observations]
        final = next((o for o in observations if o['node_id'] == terminal), None)
        final_status = final['current_condition_status'] if final else 'not_observed'
        first_flag = next((i for i, s in enumerate(states) if s == 'flagged'), None)
        clear = next((i for i, s in enumerate(states) if first_flag is not None and i > first_flag and s == 'not_flagged'), None)
        recurrent = clear is not None and 'flagged' in states[clear + 1:]
        if final_status == 'flagged':
            evolution = ('recurrent' if recurrent else 'persistent' if all(s == 'flagged' for s in states)
                         else 'late_emergent' if states[0] == 'not_flagged' else 'residual_with_observation_gaps')
        elif final_status == 'not_flagged':
            evolution = 'naturally_resolved'
        elif final_status == 'relation_absent':
            evolution = 'relation_absent_at_final'
        else:
            evolution = 'terminal_unobserved_or_unavailable'
        for observation in observations:
            observation['measurements'] = [deepcopy(references[r]['record']) for r in
                observation.get('measurement_reference_ids', []) if r in references]
            observation['diagnostic_residuals'] = [deepcopy(references[r]['record']) for r in
                observation.get('current_condition_reference_ids', []) if r in references]
            observation['relation_applicability'] = ('absent' if observation['current_condition_status'] == 'relation_absent'
                else 'observed' if observation['current_condition_status'] in ('flagged', 'not_flagged') else 'unknown')
        atoms = set(track['atom_ids'])
        preservation = [r['opportunity_id'] for r in context.get('opportunities', [])
            if r['category'] == 'regional_enhancement_assessment' and atoms.intersection(r['atom_ids'])]
        cards.append(dict(need_id='rn_' + digest(track['track_id'])[:24], track_id=track['track_id'],
            metric_id=track['metric_id'], view=track['view'], factors=track['factors'], atom_ids=track['atom_ids'],
            terminal_node_id=terminal, terminal_status=final_status, evolution=evolution,
            chemistry_transitions=[o['node_id'] for o in observations if o['status'] == 'chemical_identity_changed'],
            observations=observations, recurrent=recurrent,
            current_precursor=dict(status=states[0], evidence_ids=track.get('current_evidence_ids', []),
                measurement_ids=track.get('current_measurement_ids', []),
                coordinate_mechanism='unassessed; infer from current local measurements, not future numeric values'),
            preservation_reference_ids=preservation,
            epistemic_status='Observed native trajectory; intervention benefit and causal mechanism are untested hypotheses.',
            competing_explanations=['Native chemistry or conformation evolution', 'Observation or reference coverage change'],
            missing_observations=[o['node_id'] for o in observations if o['relation_applicability'] == 'unknown'],
            open_questions=['What necessary need remains at the explicit raw final?',
                'Which current coordinate response could change this mechanism?',
                'What independent need would this goal cover and what native benefit could it disrupt?']))
    return cards
