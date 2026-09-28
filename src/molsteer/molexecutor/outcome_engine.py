"""Outcome-aware execution: continuous guidance, chemical trials, oracle feedback."""
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from rdkit import Chem
from molsteer.common import digest,write_json,file_hash
from molsteer.molthinker.outcomes import revise_from_evidence
from .interfaces import bounded_displacement
from .outcome_reward import conflict_aware_gradient
from .oracles import VinaOracle
from .graph_search import propose,inject_hypothesis
from .chemistry import decode_endpoint,signature


def oracle_failures(candidate,base,guard):
    reasons=[]
    if candidate['vina']>base['vina']+guard['vina_regression_allowance']:reasons.append('independent_vina_regression')
    if candidate['strain']>base['strain']+guard['strain_regression_allowance']:reasons.append('independent_self_strain_regression')
    return reasons


def chemical_trials(adapter,reward,oracle,endpoint,times):
    cfg=reward.spec['discrete_search'];ledger=reward.spec['evidence_ledger']
    forces=[a for f in ledger['original_findings'] for a in f['terminal_local_forces']]
    focus=[a['atom_id'] for a in sorted(forces,key=lambda v:-v['force_kcal_mol_angstrom'])[:3]]
    focus += [a['atom_id'] for a in ledger['terminal_polar_evidence'] if a.get('buried_without_detected_polar_contact')]
    reference=reward.reference['native_final_sdf']
    if file_hash(reference['path'])!=reference['sha256']:raise ValueError('Native graph reference changed')
    candidates,enumeration=propose(endpoint,reward.vocab,focus,max_candidates=cfg['max_candidates'],
        max_changed_slots=cfg['max_changed_slots'],reference_mol=Chem.MolFromMolFile(reference['path'],removeHs=True))
    base_value,_=reward.evaluate(endpoint);base_oracle=oracle.score(endpoint)
    trials=[];winner=None
    for index,candidate in enumerate(candidates):
        row={k:v for k,v in candidate.items() if k not in ('molecule','endpoint')};row['index']=index
        state=inject_hypothesis(adapter.curr,adapter.index,candidate)
        try:
            prediction,_=adapter.predict(state=state,times=times);result=adapter.endpoint(prediction)
            failures=reward.feasible(result,endpoint)
            displacement=float((result['coords']-endpoint['coords']).norm(dim=-1).max())
            if displacement>1.:failures.append('categorical_endpoint_displacement_over_1A')
            value,detail=reward.evaluate(result);score=oracle.score(result)
            failures+=oracle_failures(score,base_oracle,reward.spec['oracle_guard'])
            # Class-token changes alone may only switch aromatic/Kekule encoding.
            changed=signature(decode_endpoint(result,reward.vocab))!=signature(decode_endpoint(endpoint,reward.vocab))
            if not changed:failures.append('no_realized_endpoint_graph_change')
            gain=float(value-base_value)
            # Frozen marginal ranking is a tie-break regularizer, not a chemical probability.
            selection_gain=gain-.01*candidate['model_ranking_cost']
            if selection_gain<=1e-5:failures.append('no_counterfactual_reward_gain')
            row.update(failures=failures,gain=gain,selection_gain=selection_gain,oracle=score,
                endpoint_displacement=displacement,actual_smiles=detail['smiles'],predicted_affinity=detail['affinity'])
            if not failures and (winner is None or selection_gain>winner['gain']):
                winner=dict(state=state,endpoint=result,index=index,gain=selection_gain)
        except (ValueError,RuntimeError) as exc:row.update(failures=[str(exc)])
        trials.append(row)
    if winner is not None:
        adapter.curr={k:v.detach() if torch.is_tensor(v) else v for k,v in winner['state'].items()}
    return dict(enumeration=enumeration,trials=trials,selected_index=winner['index'] if winner else None,
        base_oracle=base_oracle,coordinates_directly_modified=False,self_condition_directly_modified=False),winner


def run_outcome_suffix(adapter,reward,output,budget,arm):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    saved=adapter.guidance_state;initial_id=reward.spec['program_id'];start=time.perf_counter()
    if saved:
        prior=saved.get('outcome',{})
        if prior.get('initial_program_id')!=initial_id or saved['arm']!=arm:raise ValueError('Outcome restart lineage mismatch')
        reward.spec=deepcopy(prior['active_program']);reward.observables.mmff.references.update(prior['mmff_references'])
    else:prior={}
    reference=reward.reference['bindings']
    if not saved and file_hash(adapter.config['resume_checkpoint'])!=reference['origin_runtime_sha256']:
        raise ValueError('Outcome experiment must begin at its matched native runtime')
    oracle=VinaOracle(reward.reference['oracle_config'],reward.observables,reward.vocab)
    research=None
    if reward.spec.get('research',{}).get('mode') in ('shadow','active'):
        from .research_guidance import ResearchGuidance
        research=ResearchGuidance(reward.spec['research'],prior.get('research_state'))
    mask=torch.zeros_like(adapter.curr['mask'],dtype=torch.bool);mask[adapter.index]=adapter.curr['mask'][adapter.index].bool()
    path=saved.get('path_used',torch.zeros_like(mask,dtype=adapter.curr['coords'].dtype)).clone()
    commits=prior.get('categorical_commits',0);revisions=prior.get('revisions',0);conflict_streak=prior.get('conflict_streak',0)
    adapter.execution_arm=arm;rows=[];torch.cuda.reset_peak_memory_stats(adapter.device)
    write_json(output/'initial_program.json',reward.spec)
    adapter.save_stage(output,f't_{adapter.step_index/adapter.args.integration_steps:.2f}',adapter.step_index/adapter.args.integration_steps)
    interval=adapter.config.get('guidance_interval',[.5,1.])
    for step in range(adapter.step_index,adapter.args.integration_steps):
        t=step/adapter.args.integration_steps;next_t=(step+1)/adapter.args.integration_steps
        reward.set_time(float(adapter.times[0][0]));dt=adapter.grid[step+1]-adapter.grid[step]
        active=arm!='unguided' and interval[0]<=t<interval[1]
        row=dict(step=step,t=t,next_t=next_t,program_id=reward.spec['program_id'],accepted=False,categorical_commit=False)
        gradient=None
        if active:
            x=adapter.curr['coords'].detach().requires_grad_(True)
            with torch.enable_grad():
                prediction,condition=adapter.predict(coordinates=x)
                try:
                    endpoint=adapter.endpoint(prediction);parts,details=reward.components(endpoint)
                    gradient,glog=conflict_aware_gradient(parts,x,project=reward.spec['gradient_policy']['project_conflicting_components'])
                    if not torch.isfinite(gradient).all():raise ValueError('Nonfinite gradient')
                    row.update(before=details,gradient=glog)
                    conflict_streak=conflict_streak+1 if glog['affinity_structure_cosine']<reward.spec['feedback_policy']['conflict_cosine'] else 0
                except (ValueError,RuntimeError) as exc:row['unavailable']=str(exc)
        else:
            with torch.no_grad():prediction,condition=adapter.predict()
        if research is not None and active:
            row['research']=research.step(adapter,prediction,condition,dt,reward,oracle,t,next_t)
        else:adapter.native_step(prediction,condition,dt)
        del prediction,condition
        times=adapter.model._update_times(adapter.times,-1e-4) if next_t==1. else adapter.times
        reward.set_time(float(times[0][0]));policy=reward.spec['feedback_policy'];review=(step+1)%policy['review_every']==0
        with torch.no_grad():
            predicted,_=adapter.predict(times=times);base=adapter.endpoint(predicted);current=base
            row['proposal_attempts']=[]
            try:
                base_value,base_details=reward.evaluate(base)
                base_oracle=oracle.score(base) if review and active else None
                if gradient is not None:
                    delta=bounded_displacement(budget.strength*adapter.inject(gradient,float(dt)),mask,path,budget,adapter.model.coord_scale)
                    for backtrack in range(5):
                        proposal_delta=delta*.5**backtrack
                        proposal,_=adapter.predict(coordinates=adapter.curr['coords']+proposal_delta,times=times)
                        candidate=adapter.endpoint(proposal)
                        failures=reward.feasible(candidate,base)
                        try:
                            value,detail=reward.evaluate(candidate)
                            if not torch.isfinite(value) or float(value-base_value)<=1e-6:failures.append('no_same_time_reward_gain')
                            if base_oracle is not None:failures+=oracle_failures(oracle.score(candidate),base_oracle,reward.spec['oracle_guard'])
                        except (ValueError,RuntimeError) as exc:failures.append(str(exc))
                        row['proposal_attempts'].append(dict(backtrack=backtrack,failures=failures))
                        if not failures:
                            adapter.curr['coords']=adapter.curr['coords']+proposal_delta
                            path+=proposal_delta.norm(dim=-1)*adapter.model.coord_scale;current=candidate
                            row.update(accepted=True,accepted_gain=float(value-base_value),after=detail,
                                injected_max_angstrom=float(proposal_delta[adapter.index].norm(dim=-1).max())*adapter.model.coord_scale)
                            break
                search=reward.spec['discrete_search']
                if active and search['enabled'] and (step+1) in search['steps'] and commits<search['max_commits']:
                    trial,winner=chemical_trials(adapter,reward,oracle,current,times)
                    row['chemical_search']=trial
                    if winner:
                        commits+=1;current=winner['endpoint'];row['categorical_commit']=True
                event=dict(time=next_t,conflict_streak=conflict_streak)
                if review and active:
                    score=oracle.score(current);native=reward.native();row['independent_oracle']=score
                    if native.get('oracle'):
                        event.update(vina_delta=score['vina']-native['oracle']['vina'],strain_delta=score['strain']-native['oracle']['strain'],
                            affinity_delta=float(current['affinity'][reward.spec['affinity_head']])-native['affinity'][reward.spec['affinity_head']],
                            oracle=score,native_oracle=native['oracle'])
                if active and policy['enabled'] and revisions<policy['max_revisions'] and next_t<1.:
                    revised=revise_from_evidence(reward.spec,event)
                    if revised:
                        revisions+=1;conflict_streak=0;reward.spec=revised
                        write_json(output/'revisions'/f'{revised["program_id"]}.json',revised)
                        row['reward_revision']=revised['revision'];row['next_program_id']=revised['program_id']
            except (ValueError,RuntimeError) as exc:row['comparison_unavailable']=str(exc)
        row['injected_path_max_angstrom']=float(path.max());rows.append(row)
        adapter.guidance_state=dict(arm=arm,program_id=reward.spec['program_id'],path_used=path,
            outcome=dict(initial_program_id=initial_id,active_program=reward.spec,mmff_references=reward.observables.mmff.references,
                categorical_commits=commits,revisions=revisions,conflict_streak=conflict_streak,
                **({'research_state':research.state} if research is not None else {})))
        with (output/'outcome_trace.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if adapter.step_index%5==0 and next_t<1.:
            adapter.save_stage(output,f't_{next_t:.2f}',next_t);torch.save(adapter.checkpoint(),output/f'resume_t_{next_t:.2f}.pt')
        if adapter.step_index%10==0:print(json.dumps(dict(step=adapter.step_index,accepted=sum(r['accepted'] for r in rows),commits=commits,revisions=revisions)),flush=True)
    adapter.save_stage(output,'final',1.);torch.save(adapter.checkpoint(),output/'resume_final.pt')
    result=dict(arm=arm,status='complete',steps=len(rows),accepted_steps=sum(r['accepted'] for r in rows),
        categorical_commits=commits,reward_revisions=revisions,max_injected_path_angstrom=float(path.max()),
        oracle_calls=oracle.calls,wall_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),
        any_model_parameter_grad=any(p.grad is not None for p in adapter.model.parameters()),final_is_native_model_head=True,
        posthoc_coordinate_minimization=False,edited_batch_indices=[adapter.index],other_batch_items_receive_no_guidance=True,
        initial_program_id=initial_id,final_program_id=reward.spec['program_id'],
        categorical_scope='Explicit categorical current-state intervention; SC unchanged until native update; actual endpoint graph must change to commit',
        comparison_scope='Same coordinate budget; discrete trials and oracle calls add separate compute and categorical intervention budgets')
    if research is not None:
        result['research_state']=research.state
        result['categorical_scope']='Research soft probability guidance before native sampling; SC updated consistently; sampled and endpoint graph changes logged separately'
    write_json(output/'execution_summary.json',result);return result
