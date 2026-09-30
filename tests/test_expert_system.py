"""Expert handoffs, real tool loops and numerical control tests without paid APIs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from langchain_core.messages import AIMessage, ToolMessage
from molsteer.agents.config import load_config, AgentSystemConfig
from molsteer.agents.runtime import AgentRuntime
from molsteer.agents.expert_contracts import (validate_biology, validate_math, compile_expert_spec,
                                            ModelDynamicsContext, validate_expert_spec)
from molsteer.molthinker.planner import derive
from molsteer.molthinker.research.corpus import MarkdownCorpus
from molsteer.molexecutor.expert_control import control_direction, check_displacement, run_expert_trial
from molsteer.molthinker.expressions import evaluate_expression, observable_value, validate_expression
from molsteer.agents.researcher import ResearchService

ROOT=Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'examples/5i0b_A__5vef_M77/ligand_002/t_0.50'


def const(value,unit='dimensionless'):
    return {'op':'constant','value':value,'unit':unit,'origin':'Bound diagnostic reference / declared numerical-test scale'}


def op(name,*args,**extra):
    return {'op':name,'args':list(args),**extra}


@pytest.fixture
def case():
    packet,report=[json.loads((EXAMPLE/name).read_text(encoding='utf-8')) for name in ('StatePacket.json','DiagnosticReport.json')]
    term=derive(packet,report,ROOT/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md')['terms'][0]
    directions=[]
    for i,finding in enumerate(report['findings']+report['raw_state_findings']):
        selected=finding['finding_id']==term['finding_id']
        directions.append(dict(direction_id='repair' if selected else 'monitor_'+str(i),rank=i+1,
            finding_ids=[finding['finding_id']],evidence_ids=finding['evidence_ids'],
            mechanism='A localized distance deviation is a synthetic biochemical repair hypothesis.',
            evidence_class='observation',optimization_direction='Enter the evidenced distance interval.',
            repair_predicate='The selected interval violation is zero with independently checked geometry.',
            preservation_conditions=[],chemical_state='Frozen declared chemical graph for numerical validation.',
            falsifier='Independent geometry remains defective despite reduction of the distance proxy.',
            uncertainty=['Synthetic integration-test selection, not a scientific optimum.'],required=selected,
            disposition='optimize' if selected else 'monitor',priority_reason='Synthetic test singles out an evidence-bound local deviation.'))
    biology=dict(outcome='Repair a local structural defect on a coordinate copy.',
                 independent_measurement='Independent geometry and validity assessment after continuation.',
                 directions=directions,summary='Synthetic complete biological disposition for integration testing.')
    biology=validate_biology(biology,report)
    corpus=MarkdownCorpus(ROOT/'knowledge');retrieval=corpus.search('local_geometry flat-bottom distance',5)
    retrieval.update(direction_id='repair',retrieval_id='ret_fixture',provider='local')
    o={'observable_id':'distance','kind':'distance','view':term['view'],'atom_ids':term['atom_ids'],
       'evidence_ids':[term['reference_evidence_id']],'parameters':{}}
    assert len(o['atom_ids'])==2
    z={'op':'observable','id':'distance'}
    lower=op('relu',op('divide',op('subtract',const(term['lower'],'angstrom'),z),const(term['scale'],'angstrom')))
    upper=op('relu',op('divide',op('subtract',z,const(term['upper'],'angstrom')),const(term['scale'],'angstrom')))
    expression=op('multiply',const(.5),op('add',op('power',lower,exponent=2),op('power',upper,exponent=2)))
    direction=dict(direction_id='repair',status='executable',retrieval_ids=['ret_fixture'],
        source_ids=[retrieval['records'][0]['source_id']],formula='f = (relu(l-d)^2 + relu(d-u)^2)/(2*s^2)',
        derivation_summary='Specialize the retrieved flat-bottom interval by the evidenced local bounds.',
        acceptable_set='Zero exactly within the reported local distance interval.',
        normalization='One Angstrom is an explicitly uncalibrated copy-test normalization.',
        assumptions=['Frozen chemical identity and diagnostic bounds are screening references.'],
        alternatives=['A quadratic center attraction would penalize valid points inside the interval.'],
        gradient_path='Coordinate-copy derivative; live endpoint Jacobian and injection are not_run.',
        marginal_sensitivity='Linear derivative outside the interval and zero within the acceptable set.',
        failure_mode='A proxy distance can improve without removing a coupled bond-angle deformation.',
        missing_requirements=[],observables=[o],expression=expression)
    source=next(x for x in retrieval['records'] if x.get('formula'))
    direction['source_ids']=[source['source_id']]
    direction['function_lineage']=[dict(source_id=source['source_id'],locator=source['chunk_id'],
        original_formula=source['formula'],adaptation='Specialize the inspected interval form using the diagnosed distance bounds.')]
    design=dict(directions=[direction],strategy={'mode':'scalar_potential','aggregation':{'op':'single'},
                    'justification':'One sufficient synthetic target needs no cross-direction scalarization.'},
                conflict_assessment='Single copy-coordinate direction; live conflicts remain unmeasured.',
                independent_evaluation='Use independent geometry and chemical validity after a matched continuation.')
    return packet,report,biology,design,retrieval


def compiled(case):
    p,r,b,d,ret=case
    spec,defer=compile_expert_spec(p,r,b,d,[ret],ModelDynamicsContext().model_dump(),{})
    assert defer is None
    return spec


def test_versioned_spec_and_copy_derivatives(case):
    p,r,*_=case; spec=compiled(case)
    assert validate_expert_spec(spec,p,r)
    original=copy.deepcopy(p)
    trial=run_expert_trial(p,spec)
    assert trial['numerical_gradient']['passed']
    assert trial['penalty_after']<=trial['penalty_before']
    assert trial['fixed_atoms_unchanged'] and p==original
    assert trial['live_gradient']=='not_run'
    changed=copy.deepcopy(spec);changed['mathematical_design']['directions'][0]['formula']='tampered'
    with pytest.raises(ValueError,match='digest'):validate_expert_spec(changed,p,r)


def test_complete_biology_and_retrieval_coverage(case):
    p,r,b,d,ret=case
    bad=copy.deepcopy(b);bad['directions'].pop()
    with pytest.raises(ValueError):validate_biology(bad,r)
    with pytest.raises(ValueError,match='retrieval'):validate_math(d,b,p,[],{})
    bad=copy.deepcopy(d);bad['directions'][0]['source_ids']=['fabricated']
    with pytest.raises(ValueError,match='unretrieved'):validate_math(bad,b,p,[ret],{})
    bad=copy.deepcopy(d);bad['strategy']['aggregation']={'op':'sum','weights':[1.]}
    with pytest.raises(ValueError,match='weighted sums'):validate_math(bad,b,p,[ret],{})


def test_rejected_lineage_feedback_identifies_field_without_echoing_formula(case):
    from molsteer.agents.loop import _validation_feedback
    p,r,b,d,ret=case
    bad=copy.deepcopy(d)
    bad['directions'][0]['function_lineage'][0]['original_formula']='rejected-private-input'
    with pytest.raises(ValueError) as error:validate_math(bad,b,p,[ret],{})
    feedback=_validation_feedback(error.value)
    assert feedback['validation_path']==['directions',0,'function_lineage',0,'original_formula']
    assert 'exact substring' in feedback['validation_hint']
    assert 'rejected-private-input' not in json.dumps(feedback)


def test_rejected_evidence_feedback_identifies_direction_without_echoing_ids(case):
    from molsteer.agents.loop import _validation_feedback
    p,r,b,d,ret=case
    bad=copy.deepcopy(b)
    index=next(i for i,direction in enumerate(bad['directions'])
               if set(r['evidence_index'])-set(direction['evidence_ids']))
    bad['directions'][index]['evidence_ids']=[next(iter(set(r['evidence_index'])-set(bad['directions'][index]['evidence_ids'])))]
    with pytest.raises(ValueError) as error:validate_biology(bad,r)
    feedback=_validation_feedback(error.value)
    assert feedback['validation_path']==['directions',index,'evidence_ids']
    assert all(evidence_id not in json.dumps(feedback)
               for evidence_id in bad['directions'][index]['evidence_ids'])


def test_required_deferred_direction_is_not_executable(case):
    p,r,b,d,ret=case;d=copy.deepcopy(d)
    d['directions'][0].update(status='design_only',expression=None,missing_requirements=['Live physical applicability is unknown.'])
    spec,deferred=compile_expert_spec(p,r,b,d,[ret],ModelDynamicsContext().model_dump(),{})
    assert spec is None and deferred['blocked_directions']==['repair']


def test_corpus_formula_provenance_zero_hits_and_change(tmp_path):
    corpus=MarkdownCorpus(ROOT/'knowledge')
    rows=corpus.search('flat-bottom local_geometry')['records']
    assert any(r.get('formula') and r.get('variables') and r['line_start'] for r in rows)
    assert corpus.search('qzxvqqxzzz')['status']=='zero_hits'
    (tmp_path/'source.md').write_text('# Energy\n\nAn energy formula $x^2$.',encoding='utf-8')
    other=MarkdownCorpus(tmp_path);row=other.search('energy')['records'][0]
    (tmp_path/'source.md').write_text('modified',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):other.fetch(row['chunk_id'])


@pytest.mark.parametrize('mode',['scalar_potential','common_descent'])
def test_nonidentity_live_pullback_and_fixed_coordinates(mode):
    x=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
    endpoint=torch.stack([2*x[0],3*x[1],7*x[2]])
    objectives={'a':endpoint[0].square(),'b':endpoint[1].square()}
    strategy={'mode':mode,'aggregation':{'op':'lp_norm','p':2} if mode=='scalar_potential' else None}
    d,g,audit=control_direction(objectives,x,torch.tensor([1.,1.,0.]),strategy)
    assert g[0,0]==8 and g[1,1]==36 and d[2]==0
    assert audit['cosine'][0][1]==0
    assert check_displacement(.001*d,g,strategy,scalar_gradient=-d)[0]
    if mode=='common_descent':assert not check_displacement(-d,g,strategy)[0]


def test_opposition_zero_and_clipping_conflict():
    x=torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True)
    strategy={'mode':'common_descent','aggregation':None}
    d,_,audit=control_direction({'a':x[0]+1,'b':1-x[0]},x,torch.ones(2),strategy)
    assert not d.any() and audit['status']=='pareto_stationary_or_inactive'
    _,_,audit=control_direction({'a':x[0]*0+1,'b':x[1]+1},x,torch.ones(2),strategy)
    assert audit['status']=='unresolved_zero_gradient'
    # Nonuniform clipping can change the sign of an individual directional derivative.
    g=torch.tensor([[1.,-2.],[-2.,1.]])
    assert check_displacement(torch.tensor([1.,1.]),g,strategy)[0]
    assert not check_displacement(torch.tensor([.01,1.]),g,strategy)[0]


def test_expression_units_domains_and_periodicity():
    z={'op':'observable','id':'a'}
    with pytest.raises(ValueError,match='dimensions'):
        validate_expression(op('add',z,const(1,'kcal/mol')),{'a':'angstrom'})
    with pytest.raises(ValueError,match='Unsupported'):
        validate_expression({'op':'python','expression':'eval(1)'},{'a':'angstrom'})
    expr=op('power',op('divide',op('periodic_difference',z,const(-3.13,'radian')),const(1,'radian')),exponent=2)
    validate_expression(expr,{'a':'radian'})
    a=torch.tensor(3.13,dtype=torch.float64,requires_grad=True)
    assert evaluate_expression(expr,{'a':a})<.001
    with pytest.raises(ValueError,match='denominator'):
        evaluate_expression(op('divide',z,const(0)),{'a':a})


@pytest.mark.parametrize('kind,ids,params',[
    ('angle',[0,1,2],{}),('dihedral',[0,1,2,3],{}),('signed_volume',[0,1,2,3],{}),
    ('anchor_offset',[0],{'reference':[.2,.3,.4],'origin':'Test reference'}),
    ('direction_alignment',[0,1],{'reference':[1.,.3,.2],'origin':'Test direction'})])
def test_new_observable_finite_differences(kind,ids,params):
    from molsteer.molmonitor.checks import gradient_check
    x=torch.tensor([[0.,0.,0.],[1.,.1,.2],[1.8,1.,-.2],[2.,1.2,1.]],dtype=torch.float64)
    obs={'kind':kind,'atom_ids':ids,'parameters':params}
    assert gradient_check(lambda y:observable_value(obs,y,[0,1,2,3],{}),x)['passed']


class ScriptModel:
    def __init__(self,respond):self.respond=respond;self.calls=0
    def bind_tools(self,tools):self.tools={t.name for t in tools};return self
    def invoke(self,messages):
        self.calls+=1
        calls=self.respond(self.calls,messages)
        return AIMessage(content='PRIVATE_REASONING_NOT_FOR_AUDIT',tool_calls=[
            {'name':name,'args':args,'id':f'{self.calls}_{i}'} for i,(name,args) in enumerate(calls)])


def observations(messages,name):
    return [json.loads(m.content) for m in messages if isinstance(m,ToolMessage) and m.name==name]


def make_models(case,research=False,revisions=0):
    p,r,b,d,ret=case
    def biology(n,messages):
        if research and n==1:return [('request_research',{'direction_id':'repair','question':'local geometry interval repair',
            'evidence_gap':'Need the physical meaning of a flat bottom interval.','completion_condition':'Find an inspected local formula.'})]
        return [('submit_biology_plan',{'plan':b})]
    def mathematics(n,messages):
        if n<=revisions:return [('request_biology_revision',{'direction_ids':['repair'],'issue':'The selected priority requires reconsideration.',
                                                             'requested_change':'Review whether repair constraints are scientifically mandatory.'})]
        hits=observations(messages,'search_direction_knowledge')
        if not hits:
            calls=[('search_direction_knowledge',{'direction_id':'repair','query':'flat-bottom local_geometry'})]
            if research:calls.append(('request_research',{'direction_id':'repair','question':'gradient flat-bottom local_geometry',
                'evidence_gap':'Need a formula derivative and applicability limits.','completion_condition':'Inspect local function and identify its gradient target.'}))
            return calls
        design=copy.deepcopy(d);design['directions'][0]['retrieval_ids']=[hits[-1]['retrieval_id']]
        design['directions'][0]['source_ids']=[hits[-1]['records'][0]['source_id']]
        source=next(x for x in hits[-1]['records'] if x.get('formula'))
        design['directions'][0]['source_ids']=[source['source_id']]
        design['directions'][0]['function_lineage']=[dict(source_id=source['source_id'],locator=source['chunk_id'],
            original_formula=source['formula'],adaptation='Specialize the inspected interval form using the diagnosed distance bounds.')]
        return [('test_mathematical_design',{'design':design}),('submit_mathematical_design',{'design':design})]
    def researcher(n,messages):
        reads=observations(messages,'fetch_source');hits=observations(messages,'search_sources')
        if not hits:return [('search_sources',{'query':'flat-bottom local_geometry'})]
        if not reads:return [('fetch_source',{'source_id':hits[-1]['records'][0]['chunk_id']})]
        return [('submit_research',{'summary':'The inspected local reference describes a flat-bottom interval potential.',
            'claims':[{'claim':'Interval potentials can have a zero-cost accepted window.',
                       'observation_ids':[reads[-1]['observation_id']],'limitation':'Screening references require chemical applicability checks.'}], 'gaps':[]})]
    return {'molreader':ScriptModel(lambda n,m:[(k,{}) for k in ('inspect_geometry','inspect_chemistry','inspect_uncertainty','submit_diagnosis')]),
            'molthinker.biology':ScriptModel(biology),'molthinker.mathematics':ScriptModel(mathematics),
            'molthinker.researcher':ScriptModel(researcher),
            'molexecutor':ScriptModel(lambda n,m:[(k,{}) for k in ('inspect_reward_program','test_reward_program','submit_tested_program')]),
            'molmonitor':ScriptModel(lambda n,m:[(k,{}) for k in ('inspect_monitor_event','acknowledge_monitor_route')])}


def test_dual_runtime_researcher_and_checkpoint_bridge(case,tmp_path):
    from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint
    cfg=load_config();cfg.repo_root=ROOT;cfg.runtime.trace_dir=Path('outputs/test_experts');cfg.thinker.external_research=False
    models=make_models(case,research=True)
    state=AgentRuntime(cfg,models=models).run(*case[:2],run_id='dual_fixture')
    assert state['status']=='validated',state['trace'][-4:]
    assert len(state['research_packets'])==2
    assert {p['requested_by'] for p in state['research_packets']}=={'biology','mathematics'}
    assert all(p['status']=='supported' for p in state['research_packets'])
    assert 'molthinker' not in models
    assert 'PRIVATE_REASONING_NOT_FOR_AUDIT' not in json.dumps(state)
    template=json.loads((ROOT/'experiments/guidance/creativity.json').read_text())
    program,_,editable,checkpoint=compile_validated_agent_checkpoint(state['checkpoint_path'],template)
    assert program['evaluator']=='agent_expert' and editable
    assert checkpoint['artifacts']['biology_plan']==state['biology_plan']


def test_bounded_reconsideration_stops_before_executor(case):
    cfg=load_config();cfg.runtime.trace_dir=Path('outputs/test_experts');cfg.thinker.external_research=False
    models=make_models(case,revisions=3)
    state=AgentRuntime(cfg,models=models).run(*case[:2],run_id='bounded_fixture')
    assert state['status']=='design_deferred'
    assert len(state['expert_history'])==3
    assert models['molexecutor'].calls==0


def test_legacy_config_inherits_single_and_expert_model_validation():
    data=json.loads((ROOT/'configs/agents.json').read_text());data.pop('thinker')
    cfg=AgentSystemConfig.model_validate(data)
    assert cfg.thinker.architecture=='single'
    data['thinker']={'experts':{'biology':'missing'}}
    with pytest.raises(ValueError):AgentSystemConfig.model_validate(data)


def test_lineage_and_view_cannot_be_fabricated(case):
    p,r,b,d,ret=case
    bad=copy.deepcopy(d);bad['directions'][0]['function_lineage'][0]['original_formula']='unreported invented formula'
    with pytest.raises(ValueError,match='Lineage'):validate_math(bad,b,p,[ret],{})
    bad=copy.deepcopy(d);bad['directions'][0]['observables'][0]['view']='state'
    with pytest.raises(ValueError,match='representation'):validate_math(bad,b,p,[ret],{})


def test_research_cache_budget_and_api_failure_record(case):
    cfg=load_config();cfg.thinker.external_research=False;cfg.thinker.max_research_requests=1
    models=make_models(case,research=True)
    runtime=AgentRuntime(cfg,models=models)
    state={'packet':case[0],'run_id':'research_cache'}
    service=ResearchService(runtime,state)
    args=('repair','local geometry interval','Need a reported interval function.','Inspect a local formula.')
    first=service.request('biology',*args);second=service.request('mathematics',*args)
    assert first['packet_id']==second['packet_id'] and second['cache_hit']
    assert len(state['research_packets'])==1
    assert service.request('biology','different',*args[1:])['status']=='budget_exhausted'
    broken=ScriptModel(lambda n,m:(_ for _ in ()).throw(RuntimeError('SECRET_NOT_TO_EXPOSE')))
    runtime.models['molthinker.researcher']=broken
    other=ResearchService(runtime,{'packet':case[0],'run_id':'research_failed'})
    failed=other.request('biology',*args)
    assert failed['status']=='failed' and not failed['claims']
    assert 'SECRET_NOT_TO_EXPOSE' not in json.dumps(failed)


def test_independent_role_model_profiles(monkeypatch):
    from unittest.mock import Mock
    from molsteer.agents.models import create_chat_model
    import sys
    cfg=load_config()
    cfg.models['bio']=cfg.models['default'].model_copy(update={'model':'test-biology'})
    cfg.thinker.experts={'biology':'bio'}
    constructor=Mock();secrets=Mock();secrets.get.return_value='test-only'
    monkeypatch.setitem(sys.modules,'langchain_openrouter',SimpleNamespace(ChatOpenRouter=constructor))
    create_chat_model(cfg,'molthinker.biology',secrets)
    assert constructor.call_args.kwargs['model']=='test-biology'
    create_chat_model(cfg,'molthinker.mathematics',secrets)
    assert constructor.call_args.kwargs['model']==cfg.models['default'].model


def test_mmff_observable_envelope_derivative():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from molsteer.molexecutor.mmff_bridge import MMFFStrain
    mol=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(mol,randomSeed=13);AllChem.MMFFOptimizeMolecule(mol)
    mol=Chem.RemoveHs(mol);original=mol.GetConformer().GetPositions().copy()
    x=torch.tensor(original,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[2,0]+=.08
    oracle=MMFFStrain();obs={'kind':'mmff_strain','atom_ids':[0,1,2],'parameters':{}}
    fn=lambda y:observable_value(obs,y,[0,1,2],{},oracle,mol)
    value=fn(x);gradient,=torch.autograd.grad(value,x)
    delta=torch.zeros_like(x);delta[2,0]=1e-4
    finite=(fn(x+delta)-fn(x-delta))/(2e-4)
    assert float(gradient[2,0])==pytest.approx(float(finite.detach()),abs=.02)
    assert (original==mol.GetConformer().GetPositions()).all()
    assert fn(x-.0001*gradient)<value
    with pytest.raises(ValueError,match='validated stable graph'):
        observable_value(obs,x,[0,1,2],{})


def test_probe_restores_rng_coordinates_and_conditioning_on_error():
    from molsteer.molexecutor.expert_control import probe_predict
    class Adapter:
        def __init__(self):self.curr={'coords':torch.tensor([1.])};self.cond={'cached':torch.tensor([2.])}
        def predict(self,fail=False):
            draw=torch.rand(3);self.cond['cached'].add_(1);self.curr['coords'].add_(3)
            if fail:raise ValueError('Rejected probe')
            return draw,None
    adapter=Adapter();rng=torch.get_rng_state().clone()
    first,_=probe_predict(adapter);second,_=probe_predict(adapter)
    assert torch.equal(first,second) and torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError):probe_predict(adapter,fail=True)
    assert adapter.curr['coords'].item()==1 and adapter.cond['cached'].item()==2
    assert torch.equal(rng,torch.get_rng_state())


def test_numerical_failure_and_satisfied_gradients():
    from molsteer.agents.optimization import conflict_weights
    result=conflict_weights([[1.,3.],[-2.,2.],[4.,-1.]],[1,1],max_iterations=1,tolerance=1e-16)
    assert not result['converged'] and result['status']=='numerical_failure'
    x=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
    direction,_,audit=control_direction({'satisfied':x[0].square(),'active':x[1].square()},x,torch.ones_like(x),
                                       {'mode':'common_descent','aggregation':None})
    assert direction[1]<0 and audit['active_direction_ids']==['active']
    assert audit['cosine'][0][1] is None


def test_expert_live_preflight_with_mixed_views_and_nonidentity_jacobian():
    from molsteer.molmonitor.live_gradient import check_live_gradient
    class Adapter:
        index=0;precision='float64';config={'editable_atom_ids':[0,1]}
        def __init__(self):self.curr={'coords':torch.tensor([[[1.,2.,3.],[2.,3.,4.]]],dtype=torch.float64)};self.cond={}
        def predict(self,coordinates=None):return {'coords':2*coordinates[0]},{}
        def endpoint(self,pred):return pred
        def world_state_coordinates(self,x):return 3*x[0]
    class Reward:
        uses_state_view=True
        def control_gradient(self):pass
        def components(self,pred,state):return {'endpoint':pred['coords'][0].square().sum(),
                                              'state':state[1].square().sum()}
    result=check_live_gradient(Adapter(),Reward(),epsilons=(1e-5,1e-4),sample_count=3)
    assert result['passed'] and set(result['components'])=={'endpoint','state'}


def test_compiled_expert_reward_binds_live_graph_and_formula(case):
    from molsteer.molexecutor.agent_bridge import _compile_expert
    from molsteer.molexecutor.program import make_reward
    from molsteer.common import observation
    from molreader.io import load_config as vocabulary_config
    p,r,*_=case;spec=compiled(case)
    template=json.loads((ROOT/'experiments/guidance/creativity.json').read_text())
    program,_,_,_=_compile_expert(spec,p,r,{'run_id':'live_class_test'},template,1.)
    vocab=vocabulary_config();ctx=observation(p,'chemistry_context','prediction')['values'];n=len(ctx['atoms'])
    atoms=[a['element'] for a in ctx['atoms']];charges=[a['formal_charge'] for a in ctx['atoms']]
    orders=torch.zeros(n,n,dtype=torch.long)
    for bond in ctx['bonds']:
        i,j=bond['atom_ids'];orders[i,j]=orders[j,i]=vocab['bond_orders'].index(bond['bond_order'])
    pred={'coords':torch.tensor(p['steering']['coordinate_snapshots']['prediction']['coords_angstrom'],dtype=torch.float64),
          'atomics':torch.nn.functional.one_hot(torch.tensor([vocab['atomic_tokens'].index(a) for a in atoms]),len(vocab['atomic_tokens'])).double(),
          'charges':torch.nn.functional.one_hot(torch.tensor([vocab['charge_tokens'].index(a) for a in charges]),len(vocab['charge_tokens'])).double(),
          'bonds':torch.nn.functional.one_hot(orders,len(vocab['bond_orders'])).double()}
    reward=make_reward(program,pred,[],vocab)
    value,detail=reward.evaluate(pred)
    expected=run_expert_trial(p,spec,iterations=1)['penalty_before']
    assert float(value)==pytest.approx(-expected)
    x=(pred['coords']/2).clone().requires_grad_(True)
    live=dict(pred,coords=2*x)
    direction,_,_=reward.control_gradient(SimpleNamespace(),live,x,torch.ones_like(x))
    direct=torch.autograd.grad(reward.evaluate(live)[0],x)[0]
    assert torch.allclose(direction,direct)
    # Pure coordinate expressions remain mathematically evaluable after any
    # categorical change; semantic review is routed through MolMonitor.
    for key in ('atomics','charges','bonds'):
        changed=dict(pred,**{key:pred[key].roll(1,-1)})
        changed_value,_=reward.evaluate(changed)
        assert float(changed_value)==pytest.approx(float(value))
    with pytest.raises(ValueError,match='slot count'):
        reward.evaluate(dict(pred,coords=pred['coords'][:-1]))


def test_engine_rejected_expert_probes_preserve_native_path_and_rng(tmp_path,monkeypatch):
    from molsteer.molexecutor.engine import run_suffix
    from molsteer.molexecutor.interfaces import GuidanceBudget
    for name in ('reset_peak_memory_stats','max_memory_allocated'):
        monkeypatch.setattr(torch.cuda,name,lambda *args,**kwargs:0)
    class Adapter:
        index=0;batch=2;device='cpu';guidance_state={};step_index=0
        args=SimpleNamespace(integration_steps=1)
        config={};grid=torch.tensor([0.,1.]);times=[torch.zeros(2)]
        model=SimpleNamespace(coord_scale=1.,parameters=lambda:[],_update_times=lambda times,offset:times,
                              integrator=SimpleNamespace(coord_strategy='continuous',use_cosine_scheduler=False))
        def __init__(self):
            self.curr={'coords':torch.ones(2,2,3),'mask':torch.ones(2,2,dtype=torch.bool)}
            self.cond={'cache':torch.zeros(1)};self.guidance_state={};self.step_index=0
        def predict(self,coordinates=None,times=None):
            x=self.curr['coords'] if coordinates is None else coordinates
            return {'coords':2*x+torch.rand(1)*.001},self.cond
        def endpoint(self,pred):return {'coords':pred['coords'][0]}
        def inject(self,g,dt):return dt*g
        def native_step(self,pred,cond,dt):
            self.curr['coords']=self.curr['coords']+.01*torch.rand_like(self.curr['coords'])
            self.cond={'cache':torch.rand(1)};self.step_index+=1
        def save_stage(self,*args):pass
        def checkpoint(self):return {'coords':self.curr['coords'],'cond':self.cond}
    class Reward:
        spec={'program_id':'test','evaluator':'test'}
        def evaluate(self,pred):return -pred['coords'].square().sum(),{}
        def control_gradient(self,adapter,endpoint,x,mask):
            loss=endpoint['coords'].square().sum()
            direction,self.gradients,self.control_diagnostics=control_direction({'repair':loss},x,mask,
                    {'mode':'common_descent','aggregation':None})
            return direction,-loss,self.control_diagnostics
        def feasible(self,*args):return ['required_constraint_rejects']
        def proposal_failures(self,*args):return []
        def graph(self,pred):return 'same'
    rng=torch.get_rng_state().clone()
    baseline=Adapter();run_suffix(baseline,Reward(),tmp_path/'native',GuidanceBudget(),'unguided')
    end_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    controlled=Adapter();result=run_suffix(controlled,Reward(),tmp_path/'rejected',GuidanceBudget(),'expert')
    assert result['accepted_steps']==0
    assert torch.equal(controlled.curr['coords'],baseline.curr['coords'])
    assert torch.equal(controlled.cond['cache'],baseline.cond['cache'])
    assert torch.equal(torch.get_rng_state(),end_rng)


@pytest.mark.parametrize('role',['biology','mathematics'])
def test_expert_api_failure_never_falls_back(case,role):
    cfg=load_config();cfg.runtime.trace_dir=Path('outputs/test_experts')
    models=make_models(case)
    models['molthinker.'+role]=ScriptModel(lambda n,m:(_ for _ in ()).throw(RuntimeError('PRIVATE_FAILURE')))
    state=AgentRuntime(cfg,models=models).run(*case[:2],run_id='failure_'+role)
    assert state['status']=='failed' and not state['reward_spec']
    assert models['molexecutor'].calls==0 and 'PRIVATE_FAILURE' not in json.dumps(state)


def test_required_constraint_cannot_be_bought_off_by_reward_descent(case):
    p,r,b,d,ret=copy.deepcopy(case)
    biological=copy.deepcopy(next(x for x in b['directions'] if x['direction_id']=='repair'))
    biological.update(direction_id='preserve',rank=len(b['directions'])+1,disposition='constraint')
    b['directions'].append(biological)
    constraint=copy.deepcopy(d['directions'][0]);constraint['direction_id']='preserve'
    constraint['expression']=op('add',const(1),op('multiply',const(0),constraint['expression']))
    constraint['retrieval_ids']=['ret_constraint'];d['directions'].append(constraint)
    extra=dict(ret,retrieval_id='ret_constraint',direction_id='preserve')
    spec,defer=compile_expert_spec(p,r,b,d,[ret,extra],ModelDynamicsContext().model_dump(),{})
    assert defer is None
    trial=run_expert_trial(p,spec)
    assert all(not row['accepted'] for row in trial['control_trials'])
    assert trial['control_trials'][0]['constraint_failures']==['preserve']
    assert trial['penalty_before']==trial['penalty_after']
