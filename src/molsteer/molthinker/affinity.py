"""Compose an explicit task objective with an evidence-bound structural program."""
from copy import deepcopy
import math
from molsteer.common import digest


def build_affinity_program(structural_program, intent, *, retain_initial=False,
                           geometry_scope='all', affinity_weight=1.,structure_weight=1.):
    if intent.get('primary_affinity') not in ('pkd','pki','pic50','pec50'):
        raise ValueError('Declare one supported primary affinity endpoint')
    if intent.get('direction')!='maximize':
        raise ValueError('This composition requires an explicit maximize objective')
    if geometry_scope not in ('diagnosed','all') or any(not math.isfinite(w) or w<=0 for w in [affinity_weight,structure_weight]):
        raise ValueError('Invalid scope or objective weight')
    if retain_initial and (geometry_scope!='diagnosed' or structure_weight!=1.):
        raise ValueError('The retained ablation must preserve the original structural scope and weight')
    p=deepcopy(structural_program)
    p.pop('program_id',None)
    p.update(evaluator='affinity_structure',design_intent=deepcopy(intent),
        parent_program_id=structural_program['program_id'],retain_initial=retain_initial,
        geometry_scope=geometry_scope,affinity_head=intent['primary_affinity'],
        affinity_scale=1.,affinity_saturation=2.,affinity_weight=affinity_weight,
        structure_weight=structure_weight,pocket_weight=1.,pocket_contact_distance=4.5,
        pocket_distance_scale=1.,max_atom_pocket_distance=8.,
        geometry_soft_weight=.1 if geometry_scope=='all' else 0.,
        control_reference='same-time unguided trajectory from the identical full runtime checkpoint',
        reward='w_A*2*tanh((pK-pK_control(t))/2)-L_structure-pocket_penalty',
        active_objectives=['geometry','clash','pocket','movement'] if retain_initial else ['geometry','clash'],
        weights=[1.,1.,1.,1.] if retain_initial else [1.,1.],
        constraints=['sanitized_connected_guidance_proposal','no_new_or_worsened_severe_clash',
                     'pocket_occupancy','per_step_and_cumulative_injection_budget'],
        inactive_objectives=[dict(name='calibrated_affinity_uncertainty',reason='No calibrated same-endpoint error model'),
            dict(name='independent_binding_energy',reason='Not a differentiable prepared scoring pathway'),
            dict(name='torsional_forcefield_gradient',reason='Only reference-conditioned bonds/angles are differentiable here')],
        control_contract=dict(start=.5,end=1.,atom_count_change=False,
            allowed=['coordinates','element','bond_order','connectivity','formal_charge'],
            graph_reference='rebind to each sanitized predicted graph',
            categorical_control='optional deterministic probability temperature followed by native transition',
            affinity_gradient='live same-forward head to current latent coordinates',
            constraints_scope='guidance proposals; independently filter final native outputs'),
        normalization_origin='Uncalibrated experiment: one pK unit, saturation two pK units; no error-bar interpretation')
    if not retain_initial:
        p['lambda_graph']=0.
    else:
        p['constraints']=['sanitized_connected_guidance_proposal','no_new_or_worsened_severe_clash',
                         'original_endpoint_displacement_bound','per_step_and_cumulative_injection_budget']
        p['reward']='w_A*2*tanh((pK-pK_control(t))/2)+original_structural_reward'
    p['program_id']='rp_'+digest(p)[:24]
    return p
