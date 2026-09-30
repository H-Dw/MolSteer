"""Online gradient guidance with native sampling and auditable endpoint guards."""
import json
from pathlib import Path
import time
import torch
from .interfaces import GuidanceBudget, bounded_displacement
from .program import evaluate_with_state
from molsteer.molmonitor.guidance_dynamics import step_comparison, vector_cosine, linear_flow_retention
from molsteer.molmonitor.settings import monitor_settings, graph_review_settings


def run_suffix(adapter, reward, output, budget, arm):
    from functools import partial
    from .expert_control import probe_predict
    review_settings=graph_review_settings(adapter.config)
    probe=partial(probe_predict,adapter) if hasattr(reward,'control_gradient') else adapter.predict
    if review_settings and reward.spec.get('evaluator') not in ('agent_expert','agent_mixed'):
        raise ValueError('Live Agent graph review requires an Agent coordinate program; other controllers need a matching revision compiler')
    if reward.spec.get('evaluator')=='augmented_lagrangian':
        from .augmented_lagrangian_engine import run_augmented_lagrangian_suffix
        return run_augmented_lagrangian_suffix(adapter,reward,output,budget,arm)
    if reward.spec.get('evaluator')=='outcome_aware':
        from .outcome_engine import run_outcome_suffix
        return run_outcome_suffix(adapter,reward,output,budget,arm)
    if monitor_settings(adapter.config):
        from molsteer.molmonitor.runtime import run_monitored_suffix
        return run_monitored_suffix(adapter,reward,output,budget,arm)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    adapter.execution_arm=arm
    interval=adapter.config.get('guidance_interval',[0.,1.])
    if len(interval)!=2 or not 0<=interval[0]<interval[1]<=1:
        raise ValueError('Guidance interval must satisfy 0 <= start < end <= 1')
    mask=torch.zeros_like(adapter.curr['mask'],dtype=torch.bool)
    active=adapter.curr['mask'][adapter.index].bool()
    editable=adapter.config.get('editable_atom_ids')
    if editable is None:
        mask[adapter.index]=active
    else:
        if (not isinstance(editable,list) or not editable or
                any(type(i) is not int or i<0 or i>=len(active) for i in editable) or
                len(set(editable))!=len(editable) or not bool(active[editable].all())):
            raise ValueError('editable_atom_ids must name unique active target atoms')
        mask[adapter.index,editable]=True
    path_used=torch.zeros_like(mask,dtype=adapter.curr['coords'].dtype)
    if adapter.guidance_state:
        if adapter.guidance_state['arm']!=arm or adapter.guidance_state['program_id']!=reward.spec['program_id']:
            raise ValueError('Cannot change the reward or arm silently within a guided continuation')
        path_used=adapter.guidance_state['path_used'].clone()
    adapter.guidance_state.update(arm=arm,program_id=reward.spec['program_id'],path_used=path_used)
    rows=[];tensor_trace=[]
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(adapter.device)
    start_t=float(adapter.grid[adapter.step_index])
    adapter.save_stage(output,f't_{start_t:.2f}',start_t)
    for step in range(adapter.step_index,adapter.args.integration_steps):
        dt=adapter.grid[step+1]-adapter.grid[step]
        row=dict(step=step,t=float(adapter.times[0][0]),arm=arm,accepted=False)
        if hasattr(reward,'set_time'):reward.set_time(row['t'])
        guidance_active=arm!='unguided' and interval[0]<=step/adapter.args.integration_steps<interval[1]
        row['guidance_active']=guidance_active
        before_coordinates=adapter.curr['coords'].detach().clone()
        gradient=None
        if not guidance_active:
            with torch.no_grad():pred,cond=adapter.predict()
        else:
            x=adapter.curr['coords'].detach().requires_grad_(True)
            with torch.enable_grad():
                pred,cond=adapter.predict(coordinates=x)
                try:
                    endpoint=adapter.endpoint(pred)
                    if hasattr(reward,'control_gradient'):
                        gradient,value,detail=reward.control_gradient(adapter,endpoint,x,mask)
                        endpoint_gradient=None
                        row['expert_control']=detail
                    else:
                        value,detail=evaluate_with_state(reward,adapter,endpoint,x)
                        gradient,endpoint_gradient=torch.autograd.grad(value,(x,endpoint['coords']))
                    if not torch.isfinite(gradient).all():raise ValueError('Nonfinite gradient')
                    row.update(reward_before=float(value.detach()),reward_detail=detail,
                        gradient_norm=float(gradient.norm()),gradient_target='current_latent_coordinates_through_live_endpoint')
                    if endpoint_gradient is not None:
                        row['endpoint_gradient_norm']=float(endpoint_gradient.norm())
                    else:
                        row['gradient_target']='per_direction_live_pullbacks_and_declared_controller'
                    if reward.spec.get('evaluator')=='affinity_structure':
                        row['gradient_target']='current_latent_coordinates_through_live_structure_and_affinity_heads'
                        row['endpoint_gradient_scope']='structural output coordinates only; affinity can depend directly on shared hidden features'
                    elif not hasattr(reward,'control_gradient'):
                        row['pullback_norm_ratio']=float(gradient[adapter.index].norm()/endpoint_gradient.norm().clamp(min=1e-16))
                    gradient=gradient.detach()
                except ValueError as exc:
                    row['guidance_unavailable']=str(exc)
                    gradient=None
                    if hasattr(reward,'control_diagnostics'):row['expert_control']=reward.control_diagnostics
        predicted_coordinates=pred['coords'].detach().clone()
        adapter.native_step(pred,cond,dt)
        native_delta=adapter.curr['coords']-before_coordinates
        row['native_step_l2_angstrom']=float(native_delta[adapter.index].norm())*adapter.model.coord_scale
        del pred,cond
        if gradient is not None:
            raw_delta=budget.strength*adapter.inject(gradient,float(dt))*mask.unsqueeze(-1)
            delta=bounded_displacement(raw_delta,mask,path_used,budget,adapter.model.coord_scale)
            row.update(step_comparison(native_delta[adapter.index],gradient[adapter.index],raw_delta[adapter.index],delta[adapter.index],adapter.model.coord_scale))
            drift=predicted_coordinates[adapter.index]-before_coordinates[adapter.index]
            row['cosine_drift_gradient']=vector_cosine(drift,gradient[adapter.index])
            row['proposal_attempts']=[]
            with torch.no_grad():
                # Compare proposals at the SAME next time and with the same SC buffer.
                # No proposal calls consume sampler RNG or replace self-conditioning.
                comparison_times=adapter.model._update_times(adapter.times,-1e-4) if step+1==adapter.args.integration_steps else adapter.times
                if hasattr(reward,'set_time'):reward.set_time(float(comparison_times[0][0]))
                base_pred,_=probe(times=comparison_times)
                base_endpoint=adapter.endpoint(base_pred)
                try:
                    base_value,base_detail=evaluate_with_state(
                        reward,adapter,base_endpoint,adapter.curr['coords'])
                    row['next_base_reward']=float(base_value)
                    row['next_base_detail']=base_detail
                    for backtrack in range(4):
                        proposal_delta=delta*(0.5**backtrack)
                        candidate_pred,_=probe(coordinates=adapter.curr['coords']+proposal_delta,times=comparison_times)
                        candidate=adapter.endpoint(candidate_pred)
                        failures=reward.feasible(candidate,base_endpoint)
                        try:
                            if hasattr(reward,'proposal_failures'):
                                failures+=reward.proposal_failures(adapter,candidate,base_endpoint,
                                    adapter.curr['coords']+proposal_delta,adapter.curr['coords'],proposal_delta)
                            score,detail=evaluate_with_state(
                                reward,adapter,candidate,adapter.curr['coords']+proposal_delta)
                            if not torch.isfinite(score):failures.append('nonfinite_reward')
                            elif score<base_value-1e-7:failures.append('reward_regression_at_same_time')
                        except ValueError as exc:failures.append(str(exc))
                        row['proposal_attempts'].append(dict(backtrack=backtrack,failures=failures,
                            proposed_max_angstrom=float(proposal_delta[adapter.index].norm(dim=-1).max())*adapter.model.coord_scale))
                        if not failures:
                            adapter.curr['coords']=adapter.curr['coords']+proposal_delta
                            step_path=proposal_delta.norm(dim=-1)*adapter.model.coord_scale
                            path_used+=step_path
                            row.update(accepted=True,backtracks=backtrack,injected_max_angstrom=float(step_path.max()),
                                next_guided_reward=float(score),next_guided_detail=detail)
                            row['accepted_guidance_l2_angstrom']=float(proposal_delta[adapter.index].norm())*adapter.model.coord_scale
                            row['accepted_to_native_ratio']=row['accepted_guidance_l2_angstrom']/max(row['native_step_l2_angstrom'],1e-16)
                            row['same_time_reward_gain']=float(score-base_value)
                            row['candidate_graph_changed_from_base']=reward.graph(candidate)!=reward.graph(base_endpoint)
                            if step+2<len(adapter.grid) and adapter.model.integrator.coord_strategy=='continuous' and not adapter.model.integrator.use_cosine_scheduler:
                                response=candidate_pred['coords'][adapter.index]-base_pred['coords'][adapter.index]
                                row.update(linear_flow_retention(proposal_delta[adapter.index],response,
                                    float(adapter.grid[step+2]-adapter.grid[step+1]),float(adapter.times[0][0])))
                            break
                        row['rejected_constraints']=failures
                except ValueError as exc:
                    row['guard_unavailable']=str(exc)
                del base_pred,base_endpoint
            del gradient
        row['injected_path_max_angstrom']=float(path_used.max())
        adapter.guidance_state.update(arm=arm,program_id=reward.spec['program_id'],path_used=path_used)
        rows.append(row)
        if adapter.config.get('record_tensor_trace',False):
            tensor_trace.append(dict(step=step,t=row['t'],state_after=adapter.curr['coords'][adapter.index].detach().cpu().clone(),
                endpoint_before=predicted_coordinates[adapter.index].cpu(),
                atom_classes=adapter.curr['atomics'][adapter.index].argmax(-1).cpu(),
                charge_classes=adapter.curr['charges'][adapter.index].argmax(-1).cpu(),
                bond_classes=adapter.curr['bonds'][adapter.index].argmax(-1).cpu()))
        with (output/'guidance_trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if adapter.step_index in (25,50,75):
            stage_t=adapter.step_index/adapter.args.integration_steps
            stage_name=f't_{stage_t:.2f}'
            adapter.save_stage(output,stage_name,stage_t)
            torch.save(adapter.checkpoint(),output/f'resume_{stage_name}.pt')
        if adapter.step_index%10==0:
            print(json.dumps(dict(arm=arm,step=adapter.step_index,accepted=sum(r['accepted'] for r in rows))),flush=True)
    adapter.save_stage(output,'final',1.)
    torch.save(adapter.checkpoint(),output/'resume_final.pt')
    if tensor_trace:torch.save(tensor_trace,output/'tensor_trace.pt')
    result=dict(arm=arm,steps=len(rows),accepted_steps=sum(r['accepted'] for r in rows),
        final_program_id=reward.spec['program_id'],graph_reviews=0,monitor_enabled=False,
        gradient_steps=sum('gradient_norm' in r for r in rows),
        max_injected_path_angstrom=float(path_used.max()),wall_seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),
        any_model_parameter_grad=any(p.grad is not None for p in adapter.model.parameters()),
        edited_batch_indices=[adapter.index],other_batch_items_receive_no_guidance=True,
        final_is_native_model_head=True,posthoc_coordinate_minimization=False,
        constraint_scope='Guidance proposals only; native sampler chemistry and final output require independent quality assessment')
    (output/'execution_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
