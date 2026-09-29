"""Transactional guidance probes; native sampling and RNG execute once per step."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import time
import torch
from molsteer.common import write_json,file_hash
from molsteer.molexecutor.interfaces import bounded_displacement,GuidanceBudget
from molsteer.molexecutor.program import evaluate_with_state
from .features import snapshot
from .reference import ReferenceTrajectory
from .controller import MonitorPolicy,AdaptiveController
from .feedback import revision_request,compact


def guidance_capacity(norms,available,mask):
    remaining=mask&(available>1e-9)
    movable=remaining&(norms>1e-12)
    return movable,not bool(remaining.any())


def quality_sentinel(current,control,policy,head):
    actual=current.get('strain');native=control.get('strain') if control else None
    same_graph=bool(control and current.get('graph_id')==control.get('graph_id'))
    comparable=bool(actual and native and actual['converged'] and native['converged'] and same_graph)
    affinity_gain=(current['affinity'][head]-control['affinity'][head]) if control and current['affinity'].get(head) is not None and control['affinity'].get(head) is not None else None
    heavy=sum(a['element']!='H' for a in current.get('atoms',[]))
    per_atom=actual['value']/heavy if actual and actual['converged'] and heavy else None
    # A new graph is assessed against its OWN local minimum; do not compare
    # force-field total energies across graphs or silently lose this coverage.
    changed=bool(control and current.get('graph_id') and not same_graph)
    needs_review=changed and (per_atom is None or per_atom>policy.changed_graph_strain_per_heavy_atom)
    return dict(time=current['time'],comparable=comparable,actual=actual,native=native,affinity_gain=affinity_gain,
        reward_quality_conflict=bool(comparable and actual['value']>native['value']+policy.strain_allowance_kcal_mol and affinity_gain is not None and affinity_gain>.02),
        graph_changed_from_native=changed,strain_per_heavy_atom=per_atom,
        changed_graph_strain_limit=policy.changed_graph_strain_per_heavy_atom,
        changed_graph_quality_review=needs_review,
        comparison_limit='Changed graphs use a declared self-strain screen, not a cross-graph MMFF energy ranking' if changed else None)


def candidate_failures(frame,base,before,temporal,policy):
    failures=[]
    if not frame['finite']:return ['nonfinite_endpoint']
    if not frame['valid']:return ['unavailable_endpoint_checks: '+frame.get('invalid_reason','unknown')]
    if not base['valid']:return ['unavailable_native_comparison']
    worsening=frame['geometry_rms_z']>base['geometry_rms_z']+policy.geometry_worsening
    max_worse=frame['geometry_max_abs_z']>base['geometry_max_abs_z']+policy.geometry_max_worsening
    if frame['geometry_max_abs_z']>policy.severe_geometry_z and max_worse:
        failures.append('new_or_worsened_severe_geometry')
    graph_changed=frame['graph_id']!=before.get('graph_id')
    # Legal chemical identity changes are not automatically temporal failures.
    if temporal['max_ratio']>1 and (worsening or max_worse) and not graph_changed:
        failures.append('temporal_spike_with_geometry_worsening')
    return failures


def run_monitored_suffix(adapter,reward,output,budget,arm):
    from functools import partial
    from molsteer.molexecutor.expert_control import probe_predict
    probe=partial(probe_predict,adapter) if hasattr(reward,'control_gradient') else adapter.predict
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    interval=adapter.config.get('guidance_interval',[0.,1.])
    if len(interval)!=2 or not 0<=interval[0]<interval[1]<=1:
        raise ValueError('Guidance interval must satisfy 0 <= start < end <= 1')
    settings=adapter.config['monitor'];policy=MonitorPolicy(**settings.get('policy',{}))
    reference_path=Path(settings['reference']);reference=ReferenceTrajectory(json.loads(reference_path.read_text()))
    bindings=reference.bundle['bindings']
    for key in ['model_checkpoint','target_id','ligand_index','receptor']:
        expected=adapter.config['checkpoint'] if key=='model_checkpoint' else adapter.config[key]
        if bindings[key]!=expected:raise ValueError('Monitor reference binding mismatch: '+key)
    saved=adapter.guidance_state
    if saved and (saved['program_id']!=reward.spec['program_id'] or saved['arm']!=arm):
        raise ValueError('Use an explicit revision response to change a resumed reward')
    if saved.get('pending_request'):raise ValueError('Pending reward revision must be handled before continuing')
    monitor_state=saved.get('monitor',{})
    if not monitor_state and file_hash(adapter.config['resume_checkpoint'])!=bindings['origin_runtime_sha256']:
        raise ValueError('Initial monitor run must use the exact reference runtime checkpoint')
    for key,value in reference.bundle['normalization'].items():
        if reward.spec[key]!=value:raise ValueError('Reference normalization changed: '+key)
    ref_hash=file_hash(reference_path)
    if monitor_state and monitor_state['reference_sha256']!=ref_hash:raise ValueError('Monitor reference changed during continuation')
    if monitor_state and (asdict(MonitorPolicy(**monitor_state['policy']))!=asdict(policy) or monitor_state['budget']!=asdict(budget)):
        raise ValueError('Resume monitor policy/budget changed without an explicit new experiment')
    controller=AdaptiveController(policy,monitor_state.get('controller'))
    sentinel=monitor_state.get('sentinel');stopped=monitor_state.get('stopped',False)
    mask=torch.zeros_like(adapter.curr['mask'],dtype=torch.bool);mask[adapter.index]=adapter.curr['mask'][adapter.index].bool()
    if hasattr(reward,'control_gradient'):
        editable=adapter.config.get('editable_atom_ids')
        if not editable: raise ValueError('Expert monitored controls require an explicit editable mask')
        mask[adapter.index]=False;mask[adapter.index,editable]=True
    path_used=saved.get('path_used',torch.zeros_like(mask,dtype=adapter.curr['coords'].dtype)).clone()
    adapter.execution_arm=arm;rows=[];tensor_trace=[];started=time.perf_counter();pending=None
    start_t=float(adapter.grid[adapter.step_index]);adapter.save_stage(output,f't_{start_t:.2f}',start_t)
    torch.cuda.reset_peak_memory_stats(adapter.device)
    for step in range(adapter.step_index,adapter.args.integration_steps):
        t=step/adapter.args.integration_steps;next_t=(step+1)/adapter.args.integration_steps
        dt=adapter.grid[step+1]-adapter.grid[step]
        if hasattr(reward,'set_time'):reward.set_time(float(adapter.times[0][0]))
        gradient=None;unavailable=None;native_before=adapter.curr['coords'].detach().clone()
        guidance_active=not stopped and arm!='unguided' and interval[0]<=t<interval[1]
        if guidance_active:
            x=adapter.curr['coords'].detach().requires_grad_(True)
            pred,cond=adapter.predict(coordinates=x);endpoint=adapter.endpoint(pred)
            before=snapshot(endpoint,reward,t)
            try:
                if hasattr(reward,'control_gradient'):
                    gradient,value,detail=reward.control_gradient(adapter,endpoint,x,mask)
                else:
                    value,detail=evaluate_with_state(reward,adapter,endpoint,x)
                if not torch.isfinite(value):raise ValueError('nonfinite_reward')
                if not hasattr(reward,'control_gradient'):
                    gradient,=torch.autograd.grad(value,x)
                gradient=gradient.detach()
                if not torch.isfinite(gradient).all():raise ValueError('nonfinite_gradient')
            except ValueError as exc:unavailable=str(exc);gradient=None
        else:
            with torch.no_grad():pred,cond=adapter.predict();before=snapshot(adapter.endpoint(pred),reward,t)
        endpoint_coords=pred['coords'][adapter.index].detach().cpu().clone()
        adapter.native_step(pred,cond,dt)
        del pred,cond
        comparison_times=adapter.model._update_times(adapter.times,-1e-4) if next_t==1. else adapter.times
        if hasattr(reward,'set_time'):reward.set_time(float(comparison_times[0][0]))
        review=(step+1)%policy.strain_review_every==0 and next_t>=policy.strain_review_start
        trials=[];selected=None;candidate_frames={};deltas={};budget_exhausted=False
        with torch.no_grad():
            base_pred,_=probe(times=comparison_times);base_endpoint=adapter.endpoint(base_pred)
            base=snapshot(base_endpoint,reward,next_t,strain=review)
            try:
                control=reference.frame(next_t)
            except ValueError:
                control=None
            if gradient is not None and base['valid'] and reference.dense:
                try:
                    base_value,base_detail=evaluate_with_state(reward,adapter,base_endpoint,adapter.curr['coords'])
                except ValueError as exc:
                    unavailable=str(exc);gradient=None
            if gradient is not None and base['valid'] and reference.dense:
                unit=adapter.inject(gradient,float(dt))*mask.unsqueeze(-1)
                norms=unit.norm(dim=-1)*adapter.model.coord_scale
                available=torch.minimum(torch.full_like(norms,budget.max_step_angstrom),(budget.max_path_angstrom-path_used).clamp(min=0))
                movable,budget_exhausted=guidance_capacity(norms,available,mask)
                if not movable.any() and not budget_exhausted:unavailable='vanishing_coordinate_gradient'
                saturation=float((available[movable]/norms[movable]).max()) if movable.any() else policy.min_eta
                previous_deltas=[]
                for eta in (controller.strengths(saturation) if movable.any() else []):
                    delta=bounded_displacement(eta*unit,mask,path_used,budget,adapter.model.coord_scale)
                    if any(torch.allclose(delta,old,atol=1e-9,rtol=1e-6) for old in previous_deltas):continue
                    previous_deltas.append(delta)
                    candidate_pred,_=probe(coordinates=adapter.curr['coords']+delta,times=comparison_times)
                    candidate=adapter.endpoint(candidate_pred);frame=snapshot(candidate,reward,next_t,strain=review)
                    temporal=reference.temporal_evidence(before,frame,mad_scale=policy.temporal_mad_scale,
                        native_multiplier=policy.native_rate_multiplier,rate_floor=policy.rate_floor)
                    failures=reward.feasible(candidate,base_endpoint)+candidate_failures(frame,base,before,temporal,policy)
                    try:
                        if hasattr(reward,'proposal_failures'):
                            failures+=reward.proposal_failures(adapter,candidate,base_endpoint,
                                adapter.curr['coords']+delta,adapter.curr['coords'],delta)
                        score,candidate_detail=evaluate_with_state(reward,adapter,candidate,adapter.curr['coords']+delta);gain=float(score-base_value)
                        if not math.isfinite(gain):failures.append('nonfinite_reward')
                        elif gain<policy.minimum_gain:failures.append('insufficient_same_time_reward_gain')
                    except ValueError as exc:gain=None;failures.append(str(exc))
                    sf=frame.get('strain');sb=base.get('strain')
                    if sf and sb and sf['converged'] and sb['converged'] and frame.get('graph_id')==base.get('graph_id'):
                        if sf['value']>sb['value']+policy.strain_allowance_kcal_mol:
                            failures.append('same_step_strain_worsening')
                    trial=dict(eta=eta,effective_l2=float(delta[adapter.index].norm())*adapter.model.coord_scale,
                        max_step_angstrom=float(delta[adapter.index].norm(dim=-1).max())*adapter.model.coord_scale,
                        effective_eta=float(delta[adapter.index].norm()/unit[adapter.index].norm().clamp(min=1e-20)),
                        gain=gain,failures=sorted(set(failures)),temporal=temporal,
                        graph_changed=frame.get('graph_id')!=base.get('graph_id'),
                        geometry_rms_z=frame.get('geometry_rms_z'),affinity=frame.get('affinity'),strain=sf)
                    trials.append(trial);candidate_frames[eta]=frame;deltas[eta]=delta
                selected=controller.choose(trials)
                if selected:
                    delta=deltas[selected['eta']]
                    adapter.curr['coords']=adapter.curr['coords']+delta
                    path_used+=delta.norm(dim=-1)*adapter.model.coord_scale
            elif gradient is not None and not reference.dense:unavailable='sparse_reference_requires_dense_replay_for_automatic_escalation'
            elif not base['valid']:unavailable='native_endpoint_checks_unavailable: '+base.get('invalid_reason','unknown')
            current=candidate_frames[selected['eta']] if selected else base
            if review:
                sentinel=quality_sentinel(current,control,policy,reward.spec.get('affinity_head','pkd'))
        if stopped:
            decision=dict(step=step,time=next_t,destination='MolExecutor',action='stop_guidance',
                reasons=['guidance_previously_stopped'],failures=[],selected_eta=0.,selected_effective_l2=0.)
            controller.state['history']=(controller.state['history']+[decision])[-8:]
        else:
            decision=controller.route(step,next_t,trials,selected,unavailable=unavailable,sentinel=sentinel,budget_exhausted=budget_exhausted)
        if decision['action']=='stop_guidance':stopped=True
        row=dict(step=step,t=t,next_t=next_t,arm=arm,accepted=selected is not None,
            guidance_active=guidance_active,
            gradient_norm=float(gradient.norm()) if gradient is not None else None,
            gradient_target='live_reward_to_current_latent_coordinates',unavailable=unavailable,
            native_step_l2_angstrom=float((adapter.curr['coords']-native_before-(deltas[selected['eta']] if selected else 0))[adapter.index].norm())*adapter.model.coord_scale,
            selected=selected,decision=decision,proposal_attempts=trials,
            before=compact(before),native_next=compact(base),committed=compact(current),sentinel=sentinel,
            injected_path_max_angstrom=float(path_used.max()))
        rows.append(row)
        if hasattr(reward,'control_diagnostics'):
            row['expert_control']=reward.control_diagnostics
        if decision['action']=='request_reward_revision':
            pending=revision_request(reward.spec,decision,before,base,current,trials,controller.state['history'],bindings,sentinel,
                execution_state=dict(budget=asdict(budget),path_used_angstrom=path_used[adapter.index].detach().cpu().tolist(),
                    remaining_path_angstrom=(budget.max_path_angstrom-path_used[adapter.index]).clamp(min=0).detach().cpu().tolist(),
                    gradient_norm=float(gradient.norm()) if gradient is not None else None,
                    program_lineage=saved.get('program_lineage',[])))
            from molsteer.molthinker.feedback import prepare_revision_context
            context=prepare_revision_context(pending,reward.spec,settings['knowledge_path'])
            write_json(output/'feedback'/f"{pending['request_id']}.json",pending)
            write_json(output/'feedback'/f"{pending['request_id']}.context.json",context)
        adapter.guidance_state=dict(arm=arm,program_id=reward.spec['program_id'],path_used=path_used,
            program_lineage=saved.get('program_lineage',[]),pending_request=pending,
            monitor=dict(controller=controller.state,sentinel=sentinel,stopped=stopped,reference_sha256=ref_hash,policy=asdict(policy),budget=asdict(budget)))
        with (output/'monitor_trace.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        tensor_trace.append(dict(step=step,t=t,state_after=adapter.curr['coords'][adapter.index].detach().cpu().clone(),endpoint_before=endpoint_coords,
            atom_classes=adapter.curr['atomics'][adapter.index].argmax(-1).cpu(),charge_classes=adapter.curr['charges'][adapter.index].argmax(-1).cpu(),
            bond_classes=adapter.curr['bonds'][adapter.index].argmax(-1).cpu()))
        if pending:
            adapter.save_stage(output,f't_{next_t:.2f}',next_t)
            torch.save(adapter.checkpoint(),output/'resume_revision.pt');break
        if adapter.step_index%5==0 and adapter.step_index<adapter.args.integration_steps:
            adapter.save_stage(output,f't_{next_t:.2f}',next_t)
            torch.save(adapter.checkpoint(),output/f'resume_t_{next_t:.2f}.pt')
        if adapter.step_index%10==0:print(json.dumps(dict(step=adapter.step_index,eta=selected['eta'] if selected else 0.,route=decision['action'])),flush=True)
    completed=adapter.step_index==adapter.args.integration_steps and not pending
    if completed:
        adapter.save_stage(output,'final',1.);torch.save(adapter.checkpoint(),output/'resume_final.pt')
    torch.save(tensor_trace,output/'tensor_trace.pt')
    result=dict(arm=arm,status='complete' if completed else 'revision_requested',steps=len(rows),
        accepted_steps=sum(r['accepted'] for r in rows),gradient_steps=sum(r['gradient_norm'] is not None for r in rows),
        max_injected_path_angstrom=float(path_used.max()),wall_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),
        any_model_parameter_grad=any(p.grad is not None for p in adapter.model.parameters()),
        final_is_native_model_head=completed,posthoc_coordinate_minimization=False,
        edited_batch_indices=[adapter.index],other_batch_items_receive_no_guidance=True,
        revision_request_id=pending['request_id'] if pending else None,
        constraint_scope='Tested same-step proposals only; no global optimality or guaranteed terminal validity')
    write_json(output/'execution_summary.json',result)
    return result
