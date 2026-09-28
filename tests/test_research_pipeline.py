import copy,json,tempfile,unittest
from pathlib import Path
from molsteer.molthinker.research import ResearchStore,load_packet,evidence_weight
from molsteer.molthinker.research_rewards import attach_research
from molsteer.molmonitor.research_policy import initial_state,update
from molsteer.common import digest


class Provider:
    name='deterministic test fixture'
    def search(self,query,limit):
        return dict(request_url='https://example.org/fixture',raw_response='{"fixture":true}',hit_count=1,
            records=[dict(source_key='doi:fixture',title='Fixture',url='https://example.org/fixture')])


class ResearchTests(unittest.TestCase):
    def test_categorical_only_can_disable_coordinate_strength(self):
        import torch
        from molsteer.molexecutor.interfaces import GuidanceBudget,bounded_displacement
        b=GuidanceBudget(strength=0)
        delta=bounded_displacement(b.strength*torch.ones(2,3),torch.tensor([True,False]),torch.zeros(2),b)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        with self.assertRaises(ValueError):GuidanceBudget(strength=-1)

    def create(self,root):
        request=dict(subject={'target_id':'T','ligand_id':'L'},bindings={'native_reference_sha256':'native','state_packet_id':'packet'},
            capabilities=dict(variable_atom_count=False,categorical_probability_guidance=True))
        store=ResearchStore(root,request);store.search(Provider(),'query');source=next(iter(store.sources))
        card=dict(hypothesis='Fixture alternative',target='T',baseline='native',fragment_mapping={'attachment_validated':True},
            sources=[dict(url='https://example.org/fixture',title='Fixture',evidence_type='fixture',support='Fixture only',target_match=True,transformation_match=False)],
            source_ids=[source],claimed_endpoint='affinity',transfer_limits=['fixture'],counterevidence=[],expected_tradeoffs=[],
            proposed_control=dict(variables=['atomics'],atom_count_change=False),differentiable_surrogate='fixture',
            assignments=[dict(feature='atomics',slots=[0],category=1)],
            evidence_factors=dict(source_quality=1.,target_relevance=1.,transformation_support=0.,context_match=.8,counterevidence_penalty=.1))
        return store,card

    def test_deduplicated_store_and_immutable_publication(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');s.search(Provider(),'second query');self.assertEqual(len(s.sources),1)
            p=s.publish([c],{});self.assertEqual(load_packet(s.root/'ResearchPacket.json',s.request['subject'])['packet_id'],p['packet_id'])
            self.assertAlmostEqual(p['hypotheses'][0]['evidence_weight']['cap'],.18)
            with self.assertRaises(ValueError):s.publish([c],{})

    def test_raw_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');s.publish([c],{});(s.root/'raw/000.json').write_text('{}')
            with self.assertRaises(ValueError):load_packet(s.root/'ResearchPacket.json')

    def test_pending_research_can_be_resumed_for_agent_synthesis(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');pending=ResearchStore.open_pending(s.root)
            self.assertEqual(pending.request,s.request);self.assertEqual(pending.sources,s.sources)
            pending.publish([c],{'operator':'test fixture'})
            with self.assertRaises(ValueError):ResearchStore.open_pending(s.root)

    def test_claim_cannot_swap_the_retrieved_source(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');c['sources'][0]['url']='https://example.org/unretrieved'
            with self.assertRaises(ValueError):s.publish([c],{})

    def test_unknown_source_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');c['source_ids']=['missing']
            with self.assertRaises(ValueError):s.publish([c],{})

    def test_retrieval_failure_is_not_empty_positive_evidence(self):
        class Failure:
            name='failure fixture'
            def search(self,*args):raise TimeoutError('fixture timeout')
        with tempfile.TemporaryDirectory() as d:
            s=ResearchStore(Path(d)/'r',{});self.assertEqual(s.search(Failure(),'q')['status'],'failed')
            with self.assertRaises(ValueError):s.publish([],{})

    def test_optional_off_is_exact_and_active_requires_selection(self):
        p={'program_id':'unchanged'};self.assertEqual(attach_research(p,'missing',[],mode='off'),p)
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');packet=s.publish([c],{});h=packet['hypotheses'][0]['hypothesis_id']
            p=dict(evaluator='outcome_aware',packet_id='packet',native_reference={'sha256':'native'},program_id='base')
            with self.assertRaises(ValueError):attach_research(p,s.root/'ResearchPacket.json',[h],mode='active')
            active=attach_research(p,s.root/'ResearchPacket.json',[h],mode='active',allow_exploratory=True)
            self.assertEqual(active['research']['mode'],'active')
            p['packet_id']='other'
            with self.assertRaises(ValueError):attach_research(p,s.root/'ResearchPacket.json',[h],mode='active',allow_exploratory=True)

    def fixture(self):
        cfg=dict(hypotheses=[dict(hypothesis_id='h',evidence_weight={'cap':.2,'initial':.1})],category_strength=.2)
        policy=dict(bad_delta_vina=.1,bad_delta_strain=2.,good_delta_vina=-.02,patience=2,no_effect_patience=3)
        return initial_state(cfg),policy

    def test_unknown_scores_do_not_become_success(self):
        s,p=self.fixture();initial=copy.deepcopy(s)
        for _ in range(5):s,e=update(s,dict(time=.5,vina_delta=None,strain_delta=None),p)
        self.assertEqual(s['weights'],initial['weights']);self.assertEqual(e['action'],'hold')

    def test_repeated_counterevidence_routes_both_modules(self):
        s,p=self.fixture();event=dict(time=.6,vina_delta=.2,strain_delta=0.,endpoint_graph_changed=True)
        s,_=update(s,event,p);s,e=update(s,event,p)
        self.assertEqual(e['route'],'MolThinker+MolExecutor');self.assertAlmostEqual(s['weights']['h'],.05);self.assertAlmostEqual(s['strength'],.1)

    def test_only_realized_benefit_can_increase_to_cap(self):
        s,p=self.fixture()
        for _ in range(30):s,_=update(s,dict(time=.7,vina_delta=-.1,strain_delta=0.,endpoint_graph_changed=True),p)
        self.assertLessEqual(s['weights']['h'],.2);self.assertLessEqual(s['strength'],.4)

    def test_no_effect_and_fixed_ablations(self):
        s,p=self.fixture();fixed=copy.deepcopy(s);event=dict(time=.7,vina_delta=0.,strain_delta=0.,endpoint_graph_changed=False)
        for _ in range(3):s,e=update(s,event,p);fixed,_=update(fixed,event,p,dynamic=False)
        self.assertEqual(e['action'],'downweight_no_realized_effect');self.assertAlmostEqual(s['weights']['h'],.08)
        self.assertEqual(fixed['weights']['h'],.1)

    def test_nonfinite_evidence_factor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            s,c=self.create(Path(d)/'r');c['evidence_factors']['context_match']=float('nan')
            with self.assertRaises(ValueError):s.publish([c],{})


if __name__=='__main__':unittest.main()
