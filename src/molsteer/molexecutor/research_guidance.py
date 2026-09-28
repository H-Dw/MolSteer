"""Transactional categorical guidance and a resumable research evidence controller."""
from copy import deepcopy
import torch
from molsteer.common import file_hash
from molsteer.molthinker.research import load_packet
from molsteer.molmonitor.research_policy import initial_state,update
from .discrete_gradient import motif_score,guided_probabilities,apply_to_prediction
from .chemistry import decode_endpoint,signature


class ResearchGuidance:
    def __init__(self,config,saved=None):
        self.config=config;binding=config['packet']
        if file_hash(binding['path'])!=binding['sha256']:raise ValueError('Bound research packet changed')
        data=load_packet(binding['path'])
        if data['packet_id']!=binding['packet_id']:raise ValueError('Research packet identity mismatch')
        by={h['hypothesis_id']:h for h in data['hypotheses']}
        if any(by.get(h['hypothesis_id'])!=h for h in config['hypotheses']):raise ValueError('Selected hypothesis changed after research publication')
        self.state=deepcopy(saved) if saved else initial_state(config)

    def step(self,adapter,prediction,condition,dt,reward,oracle,t,next_t):
        cfg=self.config;row=dict(mode=cfg['mode'],weights=dict(self.state['weights']),strength=self.state['strength'],applied=False)
        if cfg['mode']!='active' or self.state['strength']==0 or not any(self.state['weights'].values()):
            adapter.native_step(prediction,condition,dt);row['reason']='research inactive or zero influence';return row
        if self.state['cumulative_kl']>=cfg['max_cumulative_kl']:
            adapter.native_step(prediction,condition,dt);row['reason']='categorical cumulative KL budget exhausted';return row
        keys=('atomics','bonds','charges');p={k:prediction[k][adapter.index].detach() for k in keys}
        mask={k:torch.zeros(v.shape[:-1],device=v.device,dtype=torch.bool) for k,v in p.items()}
        for h in cfg['hypotheses']:
            for term in h['assignments']:
                k=term['feature'];ids=tuple(term['slots']);mask[k][ids]=True
                if k=='bonds':mask[k][ids[::-1]]=True
        def score(values):
            return sum(self.state['weights'][h['hypothesis_id']]*motif_score(values,[h]) for h in cfg['hypotheses'])
        remaining=cfg['max_cumulative_kl']-self.state['cumulative_kl']
        q,local=guided_probabilities(p,score,mask,strength=self.state['strength'],max_kl=min(cfg['max_kl'],remaining),max_log_change=cfg['max_log_change'])
        row['soft_guidance']=local
        row['probabilities']=[dict(hypothesis=h['hypothesis_id'],feature=v['feature'],slots=v['slots'],category=v['category'],
            before=float(p[v['feature']][tuple(v['slots'])+(v['category'],)]),after=float(q[v['feature']][tuple(v['slots'])+(v['category'],)])) for h in cfg['hypotheses'] for v in h['assignments']]
        if not local['accepted']:
            adapter.native_step(prediction,condition,dt);return row
        # Compare two actual native steps using the identical incoming sampler RNG.
        incoming=adapter.checkpoint()
        adapter.native_step(prediction,condition,dt);native=adapter.checkpoint()
        times=adapter.model._update_times(adapter.times,-1e-4) if next_t==1. else adapter.times
        reward.set_time(float(times[0][0]))
        with torch.no_grad():base_pred,_=adapter.predict(times=times);base=adapter.endpoint(base_pred)
        adapter.restore(incoming)
        proposal,proposal_cond=apply_to_prediction(prediction,condition,adapter.index,q)
        adapter.native_step(proposal,proposal_cond,dt)
        with torch.no_grad():new_pred,_=adapter.predict(times=times);candidate=adapter.endpoint(new_pred)
        row['sampled_current_changes']={k:int((adapter.curr[k][adapter.index].argmax(-1)!=native['curr'][k][adapter.index].to(adapter.device).argmax(-1)).sum()) for k in keys}
        review=round(next_t*adapter.args.integration_steps)%cfg['review_every']==0
        event=dict(time=next_t,invalid=False,endpoint_graph_changed=False,vina_delta=None,strain_delta=None)
        failures=[]
        try:
            base_mol=decode_endpoint(base,reward.vocab)
            base_obs,_,_=reward.observables.evaluate(base,reward.vocab)
        except (ValueError,RuntimeError) as exc:
            failures=['native_comparator_unavailable: '+str(exc)];base_mol=None
        if base_mol is not None:
            try:
                mol=decode_endpoint(candidate,reward.vocab);changed=signature(mol)!=signature(base_mol)
                event['endpoint_graph_changed']=changed
                failures+=reward.feasible(candidate,base)
                if float((candidate['coords']-base['coords']).norm(dim=-1).max())>1.:failures.append('categorical_endpoint_displacement_over_1A')
                # Every candidate must have valid finite strain; scoring oracles
                # are additionally used on review steps and actual graph changes.
                candidate_obs,_,_=reward.observables.evaluate(candidate,reward.vocab)
                from rdkit import Chem
                event.update(strain_delta=float(candidate_obs['strain']-base_obs['strain']),
                    affinity_delta=float(candidate['affinity'][reward.spec['affinity_head']]-base['affinity'][reward.spec['affinity_head']]),
                    candidate_smiles=Chem.MolToSmiles(mol),native_smiles=Chem.MolToSmiles(base_mol))
                if event['strain_delta']>reward.spec['oracle_guard']['strain_regression_allowance']:failures.append('categorical_strain_regression')
                if review or changed:
                    co=oracle.score(candidate);bo=oracle.score(base)
                    event.update(vina_delta=co['vina']-bo['vina'],candidate_oracle=co,native_oracle=bo)
                    if event['vina_delta']>reward.spec['oracle_guard']['vina_regression_allowance']:failures.append('categorical_vina_regression')
            except (ValueError,RuntimeError) as exc:
                event['invalid']=True;failures.append(str(exc))
        row.update(failures=failures,paired_evidence=event,
            comparison='same incoming current state, self-conditioning, timestep and RNG; freshly predicted endpoints')
        if failures:
            adapter.restore(native);self.state['rejected_steps']+=1
        else:
            row['applied']=True;self.state['applied_steps']+=1
            self.state['cumulative_kl']+=local['kl_max']
            self.state['realized_current_changes']+=int(any(row['sampled_current_changes'].values()))
            self.state['realized_endpoint_changes']+=int(event['endpoint_graph_changed'])
        if review or event['invalid']:
            self.state,decision=update(self.state,event,cfg['monitor'],dynamic=cfg['dynamic']);row['monitor_decision']=decision
        row['cumulative_kl']=self.state['cumulative_kl']
        return row
