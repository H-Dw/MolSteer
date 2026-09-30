"""Chemical review routing and transactional sampler continuation (no paid APIs)."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rdkit import Chem
from molreader.io import load_config,load_stage
from molreader.packet import build_packet
from molsteer.molreader import enrich_packet
from molsteer.molexecutor.chemistry import encode_mol
from molsteer.molexecutor.interfaces import GuidanceBudget
from molsteer.molmonitor.graph_change import packet_change_event,tensor_graph,graph_changes
from molsteer.molmonitor.graph_review import GraphReviewSession,refresh_reward,live_graphs


def molecule(smiles):
    vocab=load_config();n=Chem.MolFromSmiles(smiles).GetNumAtoms()
    template=dict(coords=torch.tensor([[0.,0.,0.],[1.8,0.,0.],[2.2,1.5,0.]],dtype=torch.float64)[:n],
        atomics=torch.zeros(n,len(vocab['atomic_tokens'])),charges=torch.zeros(n,len(vocab['charge_tokens'])),
        bonds=torch.zeros(n,n,len(vocab['bond_orders'])))
    return encode_mol(Chem.MolFromSmiles(smiles),vocab,template)


def save_molecule(root,pred,time=.5):
    stage=Path(root)/'target'/'ligand_000'/f't_{time:.8f}';stage.mkdir(parents=True,exist_ok=True)
    data={k:v.unsqueeze(0) for k,v in pred.items()};data['mask']=torch.ones(1,len(pred['coords']))
    for name in ('state','structure_affinity_prediction','world_prediction'):
        torch.save(data,stage/f'{name}.pt')
    (stage/'predictions.json').write_text(json.dumps(dict(stage_t=time)),encoding='utf-8')
    return stage


def packet_at(root,pred,time=.5):
    stage=save_molecule(root,pred,time)
    contexts=[load_stage(stage,v) for v in ('state','prediction')]
    return enrich_packet(build_packet(contexts,metrics=['chemistry_context','valence','atom_inventory',
        'formal_charge','connectivity','bond_lengths','bond_angles','mmff_local_geometry']),contexts)


@pytest.mark.parametrize('smiles,head',[('COC','atomics'),('C[CH-]C','charges'),('C1CC1','bonds'),('C=CC','bonds')])
def test_independent_category_changes_trigger_review(smiles,head):
    old=molecule('CCC');new=molecule(smiles);vocab=load_config()
    changes=graph_changes({'prediction':tensor_graph(old,vocab)},{'prediction':tensor_graph(new,vocab)})
    assert changes and not changes['prediction']['slot_mapping_changed']
    assert not torch.equal(old[head],new[head])


def test_packet_changes_ignore_coordinates_and_keep_slot_roles(tmp_path):
    old=packet_at(tmp_path/'old',molecule('CCC'))
    moved=molecule('CCC');moved['coords']+=.1
    same=packet_at(tmp_path/'same',moved,.51)
    assert packet_change_event(old,same,step=51) is None
    new=packet_at(tmp_path/'new',molecule('COC'),.51)
    event=packet_change_event(old,new,step=51)
    assert event['route']=='reader'
    assert event['workflow']==['MolMonitor','MolReader','MolThinker','MolExecutor']
    assert event['changes']['prediction']['atom_changes'][0]['atom_id']==1
    assert not event['changes']['prediction']['slot_mapping_changed']


class Adapter:
    index=0;step_index=50;grid=torch.linspace(0,1,101)
    def __init__(self,pred):
        self.config={'graph_review':{'agent_config':'fixture.json','max_reviews':2},'target_id':'target'}
        self.curr={k:v.unsqueeze(0).clone() for k,v in pred.items()};self.curr['mask']=torch.ones(1,3,dtype=torch.bool)
        self.cond={'cache':torch.ones(1)}
        self.guidance_state={'arm':'agent','program_id':'old','path_used':torch.tensor([[.1,.2,.3]])}
    def predict(self,**kwargs):
        torch.rand(1)
        self.cond['cache']+=1
        return self.curr,self.cond
    def endpoint(self,pred):return {k:v[0] for k,v in pred.items() if k!='mask'}
    def checkpoint(self):
        return deepcopy(dict(curr=self.curr,cond=self.cond,guidance_state=self.guidance_state,rng=torch.get_rng_state()))
    def restore(self,cp):
        for key in ('curr','cond','guidance_state'):setattr(self,key,deepcopy(cp[key]))
        torch.set_rng_state(cp['rng'])


def reward_for(pred,program='old'):
    return SimpleNamespace(p0={k:pred[k] for k in ('atomics','charges','bonds')},vocab=load_config(),
                           uses_state_view=False,spec={'program_id':program,'evaluator':'agent_mixed'},x0=pred['coords'],receptor=[])


def test_graph_independent_state_terms_do_not_trigger_replanning(tmp_path):
    old=molecule('CCC');adapter=Adapter(old)
    adapter.predict=lambda **kwargs:({k:v.unsqueeze(0) for k,v in old.items()},adapter.cond)
    reward=reward_for(old);reward.uses_state_view=True
    reward.spec['terms']=[dict(view='prediction',graph_dependent=True),
                          dict(view='state',graph_dependent=False)]
    session=GraphReviewSession(adapter,reward,tmp_path,GuidanceBudget(),'agent',
        reviewer=lambda *a:pytest.fail('Unrelated state change triggered review'))
    # The live prediction is unchanged; only the noisy state categories move.
    adapter.curr['atomics'][0]=molecule('COC')['atomics']
    assert set(live_graphs(adapter,reward,old))=={'prediction'}
    assert session.before_step(reward)[1] is None
    reward.spec['terms'][1]['graph_dependent']=True
    assert set(live_graphs(adapter,reward,old))=={'prediction','state'}


def test_review_runs_once_and_preserves_rng_state_budget_and_resume(tmp_path):
    adapter=Adapter(molecule('COC'));reward=reward_for(molecule('CCC'));calls=[]
    def reviewer(adapter,reward,directory,event,endpoint,settings,budget):
        calls.append(event)
        torch.rand(20);adapter.curr['coords']+=100;adapter.cond['cache']+=10
        adapter.guidance_state['path_used'].zero_()
        return reward_for(endpoint,'new'),[0,1,2],{'status':'validated','strength_cap':.5}
    cp=adapter.checkpoint()
    session=GraphReviewSession(adapter,reward,tmp_path,GuidanceBudget(),'agent',reviewer=reviewer)
    revised,event,ready=session.before_step(reward)
    assert ready and event['review_result']['status']=='validated'
    torch.testing.assert_close(adapter.curr['coords'],cp['curr']['coords'])
    torch.testing.assert_close(adapter.cond['cache'],cp['cond']['cache'])
    torch.testing.assert_close(adapter.guidance_state['path_used'],cp['guidance_state']['path_used'])
    assert torch.equal(torch.get_rng_state(),cp['rng'])
    assert adapter.guidance_state['program_id']=='new' and session.strength_cap==.5
    assert session.before_step(revised)[1] is None and len(calls)==1
    session.save_state()
    resumed=GraphReviewSession(adapter,revised,tmp_path,GuidanceBudget(),'agent',reviewer=reviewer)
    assert resumed.before_step(revised)[1] is None
    assert len(calls)==1


def test_failed_review_disables_only_guidance_and_retries_next_new_graph(tmp_path):
    adapter=Adapter(molecule('COC'));reward=reward_for(molecule('CCC'));calls=[]
    def reviewer(*args):calls.append(1);raise RuntimeError('provider failure')
    cp=adapter.checkpoint()
    session=GraphReviewSession(adapter,reward,tmp_path,GuidanceBudget(),'agent',reviewer=reviewer)
    _,event,ready=session.before_step(reward)
    assert not ready and event['review_result']['status']=='review_failed'
    assert torch.equal(adapter.curr['atomics'],cp['curr']['atomics'])
    assert session.before_step(reward)[1] is None and len(calls)==1
    adapter.curr['charges'][0]=molecule('C[CH-]C')['charges']
    assert session.before_step(reward)[1] and len(calls)==2
    adapter.curr['bonds'][0]=molecule('C1CC1')['bonds']
    _,event,ready=session.before_step(reward)
    assert not ready and event['review_result']['status']=='review_budget_exhausted'
    assert len(calls)==2


def test_unguided_arm_never_activates_agents(tmp_path):
    session=GraphReviewSession(Adapter(molecule('COC')),reward_for(molecule('CCC')),tmp_path,GuidanceBudget(),'unguided',
                               reviewer=lambda *a:pytest.fail('No calls allowed'))
    assert session.before_step(reward_for(molecule('CCC')))[1:] == (None,True)


def test_live_mmff_receives_current_graph_not_source_graph():
    from molsteer.molexecutor.expert_reward import ExpertReward
    reward=ExpertReward.__new__(ExpertReward);reward.vocab=load_config();reward.uses_state_view=False
    observed=[]
    def components(coordinates,molecules):
        observed.append(Chem.MolToSmiles(molecules['prediction']))
        return {'strain':coordinates['prediction'].sum()*0}
    reward.evaluator=SimpleNamespace(mmff={'prediction':object()},
        directions=[{'observables':[{'view':'prediction'}]}],
        packet={'steering':{'coordinate_snapshots':{'prediction':{'atom_ids':[0,1,2]}}}},components=components)
    reward.components(molecule('CCC'));reward.components(molecule('COC'))
    assert observed==['CCC','COC']


def test_agent_workflow_routes_fresh_graph_through_reader_before_thinker(tmp_path):
    from molsteer.agents.runtime import AgentRuntime
    from molsteer.agents.config import load_config as agent_config
    from molreader.localized_report import make_localized_report
    from molsteer.agents.workflow import build_workflow
    from molsteer.agents.monitor import RobustMonitor
    from molsteer.agents.state import initial_state
    old=packet_at(tmp_path/'old',molecule('CCC'))
    new=packet_at(tmp_path/'new',molecule('COC'),.51)
    cfg=agent_config();cfg.mode='offline'
    runtime=AgentRuntime(cfg);runtime.monitor=RobustMonitor()
    order=[]
    # Real Reader, Monitor and LangGraph routing. Lightweight Thinker/Executor
    # stand-ins isolate handoff order from the scientific choice of new targets.
    original_reader=runtime.reader
    def reader(state):
        order.append(('reader',state['packet']['packet_id']))
        return original_reader(state)
    def thinker(state):
        order.append(('thinker',state['diagnostic_report']['packet_id']))
        state['reward_spec']={'reward_id':state['packet']['packet_id']};state['route']='executor'
        return state
    def executor(state):
        order.append(('executor',state['packet']['packet_id']))
        state['segments']+=1
        state['execution_result']=dict(done=state['segments']==2,metrics={'loss':1.},packet=new)
        state['route']='monitor';return state
    runtime.reader=reader;runtime.thinker=thinker;runtime.executor=executor
    state=initial_state(run_id='graph_route',packet=old,diagnostic_report=make_localized_report(old))
    state.update(segments=0,execute=True,strength=1.,validation_key=['old'])
    result=build_workflow(runtime).invoke(state)
    assert result['status']=='completed' and result['replans']==1
    assert order==[(who,packet) for packet in (old['packet_id'],new['packet_id']) for who in ('reader','thinker','executor')]
    assert result['packet']['representations']['prediction']['original_atom_ids']==[0,1,2]


def test_real_reader_offline_thinker_compiler_and_live_preflight(tmp_path):
    from molsteer.agents.config import load_config as agent_config
    from molsteer.molexecutor.program import MolecularReward
    from molsteer.molmonitor.graph_change import review_event
    cfg=agent_config();cfg.mode='offline';cfg.thinker.architecture='single'
    cfg.runtime.trace_dir=Path('outputs/test_graph_review')
    config_path=tmp_path/'agents.json';config_path.write_text(cfg.model_dump_json(),encoding='utf-8')
    receptor=tmp_path/'receptor.pdb'
    receptor.write_text('ATOM      1  CA  ALA A   1       0.000   5.000   0.000  1.00 20.00           C  \nEND\n',encoding='utf-8')
    class LiveAdapter(Adapter):
        precision='float64'
        def predict(self,coordinates=None,**kwargs):
            data=dict(self.curr)
            if coordinates is not None:data['coords']=coordinates
            return data,self.cond
        def save_stage(self,directory,name,time):
            save_molecule(directory,self.endpoint(self.curr),time)
        def world_state_coordinates(self,x):return x[0]
        def describe_dynamics(self):return {}
    adapter=LiveAdapter(molecule('COC'));adapter.config['receptor']=str(receptor)
    root=Path(__file__).resolve().parents[1]
    template=json.loads((root/'experiments/guidance/creativity.json').read_text(encoding='utf-8'))
    template['region_atom_ids']=[0,1,2]
    reward=MolecularReward(template,molecule('CCC'),[dict(coords=[0.,5.,0.],vdw_radius=1.7)],load_config())
    event=review_event({'prediction':{}},step=50,parent_program_id=reward.spec['program_id'])
    revised,editable,result=refresh_reward(adapter,reward,tmp_path/'review',event,molecule('COC'),
        {'agent_config':str(config_path)},GuidanceBudget())
    assert result['status']=='validated'
    assert revised.spec['evaluator']=='agent_mixed' and editable
    assert revised.spec['packet_id']!=reward.spec['packet_id']
    assert revised.spec['source_packet']['representations']['prediction']['original_atom_ids']==[0,1,2]
    torch.testing.assert_close(revised.x0,reward.x0)
    assert (tmp_path/'review'/'gradient_preflight.json').is_file()


def test_failed_review_full_suffix_matches_native_state_and_rng(tmp_path,monkeypatch):
    from molsteer.molexecutor.engine import run_suffix
    for name in ('reset_peak_memory_stats','max_memory_allocated'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**k:0)
    import molsteer.molmonitor.graph_review as module
    def fail(*args):torch.rand(30);raise RuntimeError('Unavailable Agent')
    monkeypatch.setattr(module,'refresh_reward',fail)
    class Sampler(Adapter):
        step_index=0;device='cpu';args=SimpleNamespace(integration_steps=2)
        grid=torch.tensor([0.,.5,1.]);times=[torch.zeros(1)]
        model=SimpleNamespace(coord_scale=1.,parameters=lambda:[])
        def __init__(self):
            super().__init__(molecule('COC'));self.guidance_state={}
            self.times=[torch.zeros(1)];self.step_index=0
        def native_step(self,pred,cond,dt):
            self.curr['coords']=self.curr['coords']+.01*torch.rand_like(self.curr['coords'])
            self.curr['charges']=self.curr['charges'].roll(1,-1)
            self.step_index+=1;self.times=[torch.full((1,),float(self.grid[self.step_index]))]
        def save_stage(self,*a):pass
    rng=torch.get_rng_state()
    native=Sampler();reward=reward_for(molecule('CCC'))
    run_suffix(native,reward,tmp_path/'native',GuidanceBudget(),'unguided')
    final_rng=torch.get_rng_state()
    torch.set_rng_state(rng);guided=Sampler()
    result=run_suffix(guided,reward,tmp_path/'guided',GuidanceBudget(),'agent')
    assert result['accepted_steps']==0 and result['steps']==2 and result['graph_reviews']==2
    for key in native.curr:torch.testing.assert_close(native.curr[key],guided.curr[key],atol=0,rtol=0)
    torch.testing.assert_close(native.cond['cache'],guided.cond['cache'],atol=0,rtol=0)
    assert torch.equal(final_rng,torch.get_rng_state())
