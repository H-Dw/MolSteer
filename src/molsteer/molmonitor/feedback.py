"""Compact, provenance-bound handoff from observation to reward reasoning."""
from molsteer.common import digest


def revision_request(program,decision,before,base,selected,trials,history,bindings,sentinel=None,execution_state=None):
    attempts=sorted(trials,key=lambda x:x['effective_l2'],reverse=True)
    local=[]
    for trial in attempts:
        for event in trial.get('temporal',{}).get('events',[])[:3]:
            local.append(dict(**event,eta=trial['eta'],accepted=not trial['failures']))
    request=dict(kind='RewardRevisionRequest',parent_program_id=program['program_id'],
        packet_id=program['packet_id'],identity=program['identity'],bindings=bindings,
        guard_contract={k:program[k] for k in ['severe_overlap_angstrom','max_atom_pocket_distance',
            'max_endpoint_displacement_angstrom','protein_vdw_ratio','intra_vdw_ratio','bond_tolerance_fraction','angle_tolerance_degrees'] if k in program},
        event=decision,observations=dict(
            before=compact(before),native_next=compact(base),selected=compact(selected) if selected else None,
            independent_sentinel=sentinel,localized_rate_evidence=local[:8]),
        strength_trials=[{k:t.get(k) for k in ['eta','effective_eta','effective_l2','max_step_angstrom','gain','failures','graph_changed','geometry_rms_z','affinity','strain']} for t in attempts[:9]],
        execution_state=execution_state,
        recent_history=history[-8:],
        separation=dict(observed='Values and rejected trials are measurements; missing coverage remains unknown',
            hypothesis='A reward misspecification is a hypothesis after failed strength adaptation, not established by a rate spike alone'),
        control_scope=dict(keep=['checkpoint provenance','atom mapping','coordinate frame','time convention','hard constraints','remaining path budget'],
            available=['coordinate gradient','declared categorical proposal','supported graph-conditioned terms'],
            unavailable=['calibrated affinity uncertainty','independent prepared binding scorer','variable atom count']),
        questions=['Which independent risk worsens while the optimized reward improves?',
            'Does the evidence indicate an excessive step, conflicting objectives, a missing term, or unavailable chemistry?',
            'Which proposed change is supported by the knowledge base, and how will common held-out observables test it?'],
        response_contract=dict(required=['request_id','parent_program_id','program','rationale','validation_plan'],
            resolutions=['revise_reward','stop_guidance'],
            no_silent_reset='Resume must preserve native state, conditioning, RNG and cumulative injection budget'))
    request['request_id']='rr_'+digest(request)[:24]
    return request


def compact(frame):
    if frame is None:return None
    keys=['time','view','finite','valid','chemical_valid','invalid_reason','graph_id','smiles','geometry_rms_z',
          'geometry_max_abs_z','geometry_outliers','protein_overlap','intra_overlap','affinity','strain','limitations']
    result={k:frame[k] for k in keys if k in frame}
    geometry=sorted(frame.get('geometry',{}).values(),key=lambda x:abs(x['z']),reverse=True)[:6]
    ids={i for g in geometry for i in g['atom_ids']}
    result['localized_geometry']=geometry
    result['local_atoms']=[dict(a,coords=frame['coords'][a['atom_id']]) for a in frame.get('atoms',[]) if a['atom_id'] in ids]
    return result
