"""Compile a validated Agent audit snapshot into a view-faithful FLOWR reward."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from molsteer.agents.executor import validate_and_test_reward
from molsteer.agents.trace import load_checkpoint
from molsteer.common import digest
from molsteer.molthinker.planner import validate_spec


_GUARD_FIELDS = (
    'tau', 'rho', 'graph_epsilon', 'bond_tolerance_fraction',
    'angle_tolerance_degrees', 'clash_scale_angstrom',
    'centroid_scale_angstrom', 'movement_scale_angstrom',
    'intra_vdw_ratio', 'protein_vdw_ratio', 'severe_overlap_angstrom',
    'max_endpoint_displacement_angstrom',
)
_TERM_FIELDS = (
    'term_id', 'view', 'family', 'atom_ids', 'lower', 'upper', 'scale',
    'weight', 'reference_coords', 'reference_evidence_id', 'evidence_ids',
    'unit', 'receptor_identity',
    'graph_dependent', 'hypothesis_atom_ids', 'conditions', 'observable',
)


def compile_validated_agent_checkpoint(path: str | Path, guard_template: dict | None = None):
    """Return (program, strength, editable IDs, audit snapshot).

    A checked Agent reward is not automatically a live program. This compiler
    accepts only the fixed coordinate families and preserves their declared
    state/prediction views; the caller binds a native FLOWR runtime and one weight.
    """
    checkpoint = load_checkpoint(path)
    artifacts = checkpoint.get('artifacts') or {}
    packet = artifacts.get('packet')
    report = artifacts.get('diagnostic_report')
    spec = artifacts.get('reward_spec')
    validation = artifacts.get('validation') or {}
    plan = artifacts.get('plan') or {}
    execution = artifacts.get('execution_result') or {}
    if (checkpoint.get('status') != 'validated' or execution.get('mode') != 'validation_only'
            or not validation.get('passed') or validation.get('status') != 'tested'):
        raise ValueError('Agent checkpoint must contain successful validation-only results')
    if not isinstance(packet,dict) or not isinstance(report,dict) or not isinstance(spec,dict):
        raise ValueError('Agent checkpoint lacks packet, report or reward specification')
    if (checkpoint.get('packet_sha256') != digest(packet)
            or checkpoint.get('report_sha256') != digest(report)
            or checkpoint.get('reward_id') != spec.get('reward_id')
            or validation.get('reward_id') != spec.get('reward_id')
            or plan.get('reward_id') != spec.get('reward_id')):
        raise ValueError('Agent checkpoint provenance mismatch')
    validate_spec(spec,packet,report)
    strength = artifacts.get('strength')
    if (type(strength) not in (int,float) or not math.isfinite(strength)
            or strength < 0
            or plan.get('continuation',{}).get('strength') != strength):
        raise ValueError('Agent continuation strength is invalid or inconsistent')
    if spec.get('schema_version')=='2.0.0':
        return _compile_expert(spec,packet,report,checkpoint,guard_template,strength)
    if not isinstance(guard_template,dict) or any(k not in guard_template for k in _GUARD_FIELDS):
        raise ValueError('Legacy non-expert evaluator requires its explicit physical parameter template')
    terms = spec.get('terms') or []
    if not terms or any(term.get('view') not in ('state','prediction') for term in terms):
        raise ValueError('Only state and prediction coordinate terms can run live')
    for view in {term['view'] for term in terms}:
        rep=packet['representations'][view]
        if (rep.get('coordinate_frame')!='receptor_world'
                or rep.get('coordinate_unit')!='angstrom'
                or not rep.get('transform',{}).get('verified')):
            raise ValueError('Agent term view lacks a verified receptor-world transform: '+view)
    checked=validate_and_test_reward(packet,spec,report,strength=strength)
    if not checked['passed'] or checked['status']!='tested':
        raise ValueError('Agent reward failed fresh numerical validation')
    compiled_terms=[]
    for term in terms:
        selected={k:deepcopy(term[k]) for k in _TERM_FIELDS if k in term}
        selected['source_term_digest']=digest(term)
        compiled_terms.append(selected)
    editable=sorted({atom for term in terms for atom in term['atom_ids']})
    designed='design' in spec
    program={
        'kind':'RewardProgram', 'mode':'agent_design' if designed else 'agent_selection', 'evaluator':'agent_mixed',
        'packet_id':packet['packet_id'], 'identity':deepcopy(packet['identity']),
        'source_reward_id':spec['reward_id'], 'agent_run_id':checkpoint['run_id'],
        'source_packet_digest':digest(packet), 'source_report_digest':digest(report),
        'source_packet':deepcopy(packet),
        'terms':compiled_terms, 'region_atom_ids':editable,
        'active_objectives':[], 'weights':[], 'lambda_graph':0.,
        'reward':('negative validated declarative objective tree' if designed else
                  'negative sum of view-bound dimensionless interval penalties'),
        'graph_policy':'Native graph may change; reject invalid candidate endpoints',
        'constraints':['validated source evidence','editable atom mask','gradient screen',
                       'same-time proposal guards','per-step and path budget'],
        'knowledge_source':deepcopy(spec.get('knowledge_source')),
        'retrieval':deepcopy(spec.get('retrieval',[])),
        'runtime_mapping':'Host FLOWR adapter supplies full checkpoint, editable mask and bounded strength',
    }
    if designed:
        program['design']=deepcopy(spec['design'])
        program['inactive_compatibility_fields']=['tau','rho','lambda_graph','weights']
    program.update({k:deepcopy(guard_template[k]) for k in _GUARD_FIELDS})
    program['program_id']='rp_'+digest(program)[:24]
    return program,float(strength),editable,checkpoint


__all__=['compile_validated_agent_checkpoint']


def _compile_expert(spec,packet,report,checkpoint,guard_template,strength):
    checked=validate_and_test_reward(packet,spec,report,strength=strength)
    if not checked['passed']: raise ValueError('Expert reward failed fresh validation')
    observables=[o for d in spec['mathematical_design']['directions'] if d['status']=='executable' for o in d['observables']]
    for view in {o['view'] for o in observables}:
        rep=packet['representations'][view]
        if (rep['coordinate_frame']!='receptor_world' or rep['coordinate_unit']!='angstrom'
                or not rep['transform'].get('verified')
                or rep['original_atom_ids']!=list(range(len(rep['original_atom_ids'])))):
            raise ValueError('Expert FLOWR views require verified world transforms and native slot mapping')
    editable=list(packet['representations']['prediction']['original_atom_ids'])
    declared=spec['model_dynamics']['editable_atom_ids']
    if declared is not None: editable=sorted(set(editable)&set(declared))
    if not editable: raise ValueError('No editable expert coordinates')
    program=dict(kind='RewardProgram',mode='agent_expert',evaluator='agent_expert',
                 packet_id=packet['packet_id'],identity=deepcopy(packet['identity']),
                 source_reward_id=spec['reward_id'],agent_run_id=checkpoint['run_id'],
                 expert_spec=deepcopy(spec),source_packet=deepcopy(packet),source_report=deepcopy(report),
                 region_atom_ids=editable,
                 knowledge_source={'corpus':'knowledge/','retrieval':deepcopy(spec['retrieval'])},
                 graph_policy='Interpret declared mechanisms using current chemistry; no initial whole-graph activation check',
                 execution_semantics='native_scalar_gradient', guidance_weight=float(strength),
                 fixed_atom_ids=sorted(set(packet['representations']['prediction']['original_atom_ids'])-set(editable)),
                 ignored_legacy_fields=sorted(guard_template or {}))
    program['program_id']='rp_'+digest(program)[:24]
    return program,float(strength),editable,checkpoint
