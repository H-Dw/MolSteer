"""FLOWR suffix execution for the staged local-first reward."""
import json
import math
from pathlib import Path
import time

import torch

from molsteer.common import write_json
from .interfaces import bounded_displacement
from .oracles import VinaOracle


def _oracle_failures(candidate,base,guard):
    failures=[]
    if candidate['vina']>base['vina']+guard['vina_regression_allowance']:failures.append('independent_vina_regression')
    if candidate['strain']>base['strain']+guard['strain_regression_allowance']:failures.append('independent_global_strain_regression')
    return failures


def run_local_first_suffix(adapter,reward,output,budget,arm='local_first'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    saved=adapter.guidance_state
    if saved:
        if saved.get('arm')!=arm:raise ValueError('Local-first restart lineage mismatch')
        reward.restore_controller(saved.get('local_first_controller',{}))
    reference=reward.reference['bindings']
    from molsteer.common import file_hash
    if not saved and file_hash(adapter.config['resume_checkpoint'])!=reference['origin_runtime_sha256']:
        raise ValueError('Local-first experiment must begin at its matched native runtime')
    oracle=VinaOracle(reward.reference['oracle_config'],reward.observables,reward.vocab)
    path=saved.get('path_used',torch.zeros_like(adapter.curr['mask'],dtype=adapter.curr['coords'].dtype)).clone()
    adapter.execution_arm=arm;rows=[];torch.cuda.reset_peak_memory_stats(adapter.device)
    adapter.save_stage(output,f't_{adapter.step_index/adapter.args.integration_steps:.2f}',adapter.step_index/adapter.args.integration_steps)
    interval=adapter.config.get('guidance_interval',[.5,1.]);review_every=reward.local_cfg['review_every']
    for step in range(adapter.step_index,adapter.args.integration_steps):
        t=step/adapter.args.integration_steps;next_t=(step+1)/adapter.args.integration_steps
        reward.set_time(float(adapter.times[0][0]));dt=adapter.grid[step+1]-adapter.grid[step];active=interval[0]<=t<interval[1]
        row=dict(step=step,t=t,next_t=next_t,accepted=False,active=active)
        gradient=None;editable=None
        if active:
            x=adapter.curr['coords'].detach().requires_grad_(True)
            with torch.enable_grad():
                prediction,condition=adapter.predict(coordinates=x)
                try:
                    endpoint=adapter.endpoint(prediction);row['graph_before']=reward.observe_graph(endpoint,reward.time)
                    editable=reward.editable_mask(endpoint);gradient,glog=reward.gradient(endpoint,x);row['gradient']=glog
                except (ValueError,RuntimeError) as exc:row['unavailable']=str(exc)
        else:
            with torch.no_grad():prediction,condition=adapter.predict()
        adapter.native_step(prediction,condition,dt);del prediction,condition
        times=adapter.model._update_times(adapter.times,-1e-4) if next_t==1. else adapter.times
        reward.set_time(float(times[0][0]));review=(step+1)%review_every==0 or next_t==1.
        with torch.no_grad():
            predicted,_=adapter.predict(times=times);base=adapter.endpoint(predicted)
            try:row['graph_after_native']=reward.observe_graph(base,reward.time)
            except (ValueError,RuntimeError) as exc:row['graph_after_native_unavailable']=str(exc)
            try:
                _,base_detail=reward.evaluate(base);row['native_proposal_endpoint']=base_detail;row['proposal_attempts']=[]
                base_oracle=oracle.score(base) if review and active else None
                if gradient is not None:
                    mask=torch.zeros_like(adapter.curr['mask'],dtype=torch.bool);mask[adapter.index]=editable
                    delta=bounded_displacement(budget.strength*adapter.inject(gradient,float(dt)),mask,path,budget,adapter.model.coord_scale)
                    for backtrack in range(5):
                        proposal_delta=delta*.5**backtrack
                        proposal,_=adapter.predict(coordinates=adapter.curr['coords']+proposal_delta,times=times)
                        candidate=adapter.endpoint(proposal);failures=reward.feasible(candidate,base)
                        changed=candidate['atomics'].argmax(-1).ne(base['atomics'].argmax(-1)).any() or candidate['charges'].argmax(-1).ne(base['charges'].argmax(-1)).any() or candidate['bonds'].argmax(-1).ne(base['bonds'].argmax(-1)).any()
                        candidate_oracle=None
                        if (review or bool(changed)) and not failures:
                            candidate_oracle=oracle.score(candidate);comparison=base_oracle or oracle.score(base)
                            failures+=_oracle_failures(candidate_oracle,comparison,reward.local_cfg['oracle_guard'])
                        accepted,gain,reason,candidate_detail,_=reward.compare(candidate,base)
                        if not accepted:failures.append(reason)
                        logged_gain=gain if math.isfinite(gain) else None
                        row['proposal_attempts'].append(dict(backtrack=backtrack,failures=failures,gain=logged_gain,
                            stage=candidate_detail['stage'],graph_changed=bool(changed),oracle=candidate_oracle))
                        if not failures:
                            adapter.curr['coords']=adapter.curr['coords']+proposal_delta
                            path+=proposal_delta.norm(dim=-1)*adapter.model.coord_scale
                            row.update(accepted=True,accepted_gain=gain,after=candidate_detail,
                                injected_max_angstrom=float(proposal_delta[adapter.index].norm(dim=-1).max())*adapter.model.coord_scale)
                            break
                if review:
                    with torch.no_grad():final_pred,_=adapter.predict(times=times);current=adapter.endpoint(final_pred)
                    row['review_oracle']=oracle.score(current);_,row['review_detail']=reward.evaluate(current)
            except (ValueError,RuntimeError) as exc:row['comparison_unavailable']=str(exc)
        row['injected_path_max_angstrom']=float(path.max());rows.append(row)
        adapter.guidance_state=dict(arm=arm,path_used=path,local_first_controller=reward.controller_state())
        with (output/'local_first_trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if adapter.step_index%5==0 and next_t<1.:
            adapter.save_stage(output,f't_{next_t:.2f}',next_t);torch.save(adapter.checkpoint(),output/f'resume_t_{next_t:.2f}.pt')
        if adapter.step_index%10==0:print(json.dumps(dict(step=adapter.step_index,accepted=sum(r['accepted'] for r in rows),stage=row.get('review_detail',{}).get('stage'))),flush=True)
    adapter.save_stage(output,'final',1.);torch.save(adapter.checkpoint(),output/'resume_final.pt')
    stages={}
    for row in rows:
        stage=row.get('gradient',{}).get('stage','unavailable');stages[stage]=stages.get(stage,0)+1
    result=dict(arm=arm,status='complete',steps=len(rows),accepted_steps=sum(r['accepted'] for r in rows),stage_counts=stages,
        max_injected_path_angstrom=float(path.max()),oracle_calls=oracle.calls,wall_seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),any_model_parameter_grad=any(p.grad is not None for p in adapter.model.parameters()),
        final_is_native_model_head=True,posthoc_coordinate_minimization=False,edited_batch_indices=[adapter.index],
        controller=reward.controller_state(),reward_architecture='branch gate -> conditional local geometry -> local relaxation strain -> projected affinity')
    write_json(output/'execution_summary.json',result);return result
