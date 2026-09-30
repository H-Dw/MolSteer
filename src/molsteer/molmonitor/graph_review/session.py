"""Optional live MolReader -> MolThinker -> MolExecutor graph review.

The sampler keeps its slots, state, RNG and cumulative displacement budget.
Only a validated new reward is installed. Failed/deferred review suspends extra
guidance; native categorical sampling continues, including further graph changes.
"""
from copy import deepcopy
from pathlib import Path
import torch
from molsteer.common import write_json,digest
from molsteer.molexecutor.expert_control import probe_predict
from .changes import tensor_graph,graph_changes,review_event
from ..settings import graph_review_settings


def graph_dependent_views(reward):
    """Only chemistry-dependent terms need a new chemical interpretation."""
    terms=reward.spec.get('terms')
    if isinstance(terms,list):
        views={term['view'] for term in terms if term.get('graph_dependent',True)}
    else:
        views={'prediction'}
        if getattr(reward,'uses_state_view',False):views.add('state')
    if float(reward.spec.get('lambda_graph',0))>0:views.add('prediction')
    return views


def live_graphs(adapter,reward,endpoint):
    views=graph_dependent_views(reward)
    graphs={}
    if 'prediction' in views:graphs['prediction']=tensor_graph(endpoint,reward.vocab)
    if 'state' in views:
        current={k:adapter.curr[k][adapter.index] for k in ('atomics','charges','bonds')}
        graphs['state']=tensor_graph(current,reward.vocab)
    return graphs


def refresh_reward(adapter,reward,directory,event,endpoint,settings,budget):
    """Run the actual configured Agent workflow, then compile and preflight it."""
    from molreader.io import load_stage
    from molreader.packet import build_packet
    from molsteer.molreader import enrich_packet
    from molsteer.agents.config import load_config
    from molsteer.agents.runtime import AgentRuntime
    from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint
    from molsteer.molexecutor.program import make_reward
    from ..live_gradient import check_live_gradient
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    time=float(adapter.grid[adapter.step_index])
    name=f't_{time:.8f}'
    adapter.save_stage(directory,name,time)
    stage=directory/adapter.config['target_id']/f'ligand_{adapter.index:03d}'/name
    torch.save(adapter.checkpoint(),stage/'runtime.pt')
    contexts=[load_stage(stage,view,receptor=adapter.config['receptor']) for view in ('state','prediction')]
    packet=enrich_packet(build_packet(contexts),contexts)
    write_json(directory/'StatePacket.json',packet)
    # Original slot IDs are immutable even though their chemistry changes.
    for representation in packet['representations'].values():
        if representation['original_atom_ids']!=list(range(len(endpoint['coords']))):
            raise ValueError('Graph review requires an explicit native slot mapping')
    # Re-reading must describe the exact hypothesis that activated this event.
    predicted=next(c for c in contexts if c.view=='prediction')
    observed=tensor_graph({k:torch.as_tensor(v) for k,v in predicted.probs.items()},reward.vocab)
    if observed!=tensor_graph(endpoint,reward.vocab):
        raise ValueError('Exported graph differs from the triggering live prediction')
    config=load_config(settings['agent_config'])
    config.runtime.trace_dir=config.runtime.trace_dir/'graph_reviews'/digest(str(directory.resolve()))[:16]
    feedback=deepcopy(event)
    feedback['previous_reward']=deepcopy(reward.spec)
    feedback['remaining_path_angstrom']=(budget.max_path_angstrom-
        adapter.guidance_state['path_used'][adapter.index]).clamp(min=0).detach().cpu().tolist()
    dynamics=adapter.describe_dynamics()
    # Previously selected reward support is not a permanent chemical-role mask.
    # Only a separately declared host restriction limits a new slot allocation.
    dynamics['editable_atom_ids']=adapter.config.get('host_editable_atom_ids')
    runtime=AgentRuntime(config,knowledge_path=settings.get('knowledge_path'),model_dynamics=dynamics)
    result=runtime.run(packet,run_id=event['event_id'],execute=False,feedback=feedback)
    if result['status']!='validated':
        return None,None,dict(status=result['status'],checkpoint_path=result['checkpoint_path'])
    program,strength,editable,_=compile_validated_agent_checkpoint(result['checkpoint_path'],reward.spec)
    allowed=adapter.config.get('host_editable_atom_ids')
    if allowed is not None:editable=sorted(set(editable)&set(allowed))
    active=adapter.curr['mask'][adapter.index].bool()
    if not editable or any(i<0 or i>=len(active) or not bool(active[i]) for i in editable):
        raise ValueError('Revised reward has no valid editable native slots')
    new_reward=make_reward(program,endpoint,reward.receptor,reward.vocab)
    # Do not reset coordinate trust anchors or the cumulative path on a rebind.
    new_reward.x0=reward.x0.detach().clone()
    old_editable=adapter.config.get('editable_atom_ids')
    adapter.config['editable_atom_ids']=editable
    try:preflight=check_live_gradient(adapter,new_reward)
    finally:
        if old_editable is None:adapter.config.pop('editable_atom_ids',None)
        else:adapter.config['editable_atom_ids']=old_editable
    write_json(directory/'gradient_preflight.json',preflight)
    write_json(directory/'RewardProgram.json',program)
    if not preflight['passed']:return None,None,dict(status='live_preflight_failed')
    return new_reward,editable,dict(status='validated',checkpoint_path=result['checkpoint_path'],
        program_id=program['program_id'],packet_id=packet['packet_id'],
        strength_cap=min(float(strength),budget.strength))


class GraphReviewSession:
    """Opt-in MolMonitor child; detect each newly observed graph once."""
    def __init__(self,adapter,reward,output,budget,arm,reviewer=None):
        settings=graph_review_settings(adapter.config)
        self.enabled=arm!='unguided' and bool(settings)
        self.adapter=adapter;self.output=Path(output);self.budget=budget
        self.settings=settings or {};self.reviewer=reviewer or refresh_reward
        self.state=deepcopy(adapter.guidance_state.get('graph_review',{}))
        if not self.enabled:return
        if reward.spec.get('evaluator') not in ('agent_expert','agent_mixed'):
            raise ValueError('Graph review requires an Agent coordinate program')
        if not self.settings.get('agent_config'):
            raise ValueError('graph_review requires an explicit Agent configuration')
        limit=self.settings.get('max_reviews',4)
        if type(limit) is not int or limit<1:raise ValueError('max_reviews must be a positive integer')
        if self.state and self.state.get('settings')!=self.settings:
            raise ValueError('Cannot silently change resumed graph review settings')
        self.state.setdefault('settings',deepcopy(self.settings))
        self.state.setdefault('reviews',0);self.state.setdefault('guidance_ready',True)
        self.state.setdefault('strength_cap',budget.strength)
        if 'graphs' not in self.state:
            views=graph_dependent_views(reward)
            graphs={}
            if 'prediction' in views:graphs['prediction']=tensor_graph(reward.p0,reward.vocab)
            # Compare state only once a live observation is available. Subsequent
            # state changes are detected independently of predicted endpoints.
            if 'state' in views:
                from molsteer.common import observation
                context=observation(reward.spec.get('source_packet',{}),'chemistry_context','state') if reward.spec.get('source_packet') else None
                if context:
                    atoms=context['values']['atoms'];n=len(atoms)
                    orders=torch.zeros(n,n,dtype=torch.long)
                    for bond in context['values']['bonds']:
                        i,j=bond['atom_ids'];orders[i,j]=orders[j,i]=reward.vocab['bond_orders'].index(bond['bond_order'])
                    def onehot(values,key):return torch.nn.functional.one_hot(torch.as_tensor(values),len(reward.vocab[key])).float()
                    current=dict(atomics=onehot([reward.vocab['atomic_tokens'].index(a['element']) for a in atoms],'atomic_tokens'),
                        charges=onehot([reward.vocab['charge_tokens'].index(a['formal_charge']) for a in atoms],'charge_tokens'),
                        bonds=onehot(orders,'bond_orders'))
                else:current={k:adapter.curr[k][adapter.index] for k in ('atomics','charges','bonds')}
                graphs['state']=tensor_graph(current,reward.vocab)
            self.state['graphs']=graphs

    def before_step(self,reward):
        if not self.enabled:return reward,None,True
        adapter=self.adapter
        with torch.no_grad():
            prediction,_=probe_predict(adapter)
            endpoint=adapter.endpoint(prediction)
            graphs=live_graphs(adapter,reward,endpoint)
        changes=graph_changes(self.state['graphs'],graphs)
        if not changes:return reward,None,self.state['guidance_ready']
        event=review_event(changes,step=adapter.step_index,parent_program_id=reward.spec['program_id'])
        self.state['graphs']=deepcopy(graphs)
        if self.state['reviews']>=self.settings.get('max_reviews',4):
            event['review_result']={'status':'review_budget_exhausted'}
            self.state['guidance_ready']=False
        else:
            self.state['reviews']+=1
            # Reader export, Agent code and numerical probes must not consume
            # native randomness or mutate sampling/conditioning tensors.
            checkpoint=adapter.checkpoint()
            directory=self.output/'graph_reviews'/f"{adapter.step_index:04d}_{self.state['reviews']:03d}"
            try:
                new_reward,editable,result=self.reviewer(adapter,reward,directory,event,endpoint,self.settings,self.budget)
            except Exception as exc:
                new_reward=None;editable=None
                result=dict(status='review_failed',error_type=type(exc).__name__)
            finally:adapter.restore(checkpoint)
            event['review_result']=result
            self.state['guidance_ready']=new_reward is not None and result.get('status')=='validated'
            if self.state['guidance_ready']:
                parent=reward.spec['program_id'];reward=new_reward
                adapter.config['editable_atom_ids']=editable
                self.state['strength_cap']=min(self.state['strength_cap'],result.get('strength_cap',self.budget.strength))
                # Source views can change after redesign; record the binding that
                # was actually reviewed, so the next unchanged step is not a loop.
                self.state['graphs']=live_graphs(adapter,reward,endpoint)
                adapter.guidance_state['program_id']=reward.spec['program_id']
                adapter.guidance_state.setdefault('program_lineage',[]).append(dict(
                    parent=parent,program_id=reward.spec['program_id'],event_id=event['event_id']))
                self.state['program']=deepcopy(reward.spec)
                self.state['editable_atom_ids']=list(editable)
        self.state['last_event']=deepcopy(event)
        adapter.guidance_state['graph_review']=deepcopy(self.state)
        write_json(self.output/'graph_reviews'/f"{event['event_id']}.json",event)
        return reward,event,self.state['guidance_ready']

    def save_state(self):
        if self.enabled:self.adapter.guidance_state['graph_review']=deepcopy(self.state)

    def needs_review(self,reward,endpoint):
        return self.enabled and bool(graph_changes(self.state['graphs'],live_graphs(self.adapter,reward,endpoint)))

    @property
    def strength_cap(self):
        return self.state.get('strength_cap',self.budget.strength)
