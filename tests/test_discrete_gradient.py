import unittest
import torch
from molsteer.molexecutor.discrete_gradient import guided_probabilities,motif_score,apply_to_prediction
from molsteer.molthinker.researcher import assess_hypothesis


class DiscreteGradientTests(unittest.TestCase):
    def fixture(self):
        p={'atomics':torch.tensor([[.7,.3],[.8,.2]],dtype=torch.double),'bonds':torch.tensor([[[1.,0.],[.6,.4]],[[.6,.4],[1.,0.]]],dtype=torch.double)}
        mask={'atomics':torch.tensor([True,False]),'bonds':torch.tensor([[False,True],[True,False]])}
        h=[dict(assignments=[dict(feature='atomics',slots=[0],category=1),dict(feature='bonds',slots=[0,1],category=1)])]
        return p,mask,lambda v:motif_score(v,h)

    def test_true_derivative_and_trust_region(self):
        p,m,s=self.fixture();q={k:v.clone().requires_grad_(True) for k,v in p.items()}
        g=torch.autograd.grad(s(q),q['atomics'])[0];eps=1e-6
        plus={k:v.clone() for k,v in p.items()};minus={k:v.clone() for k,v in p.items()}
        plus['atomics'][0,1]+=eps;minus['atomics'][0,1]-=eps
        self.assertAlmostEqual(float(g[0,1]),float((s(plus)-s(minus))/(2*eps)),places=6)
        out,log=guided_probabilities(p,s,m,strength=30,max_kl=.01)
        self.assertTrue(log['accepted']);self.assertLessEqual(log['kl_max'],.01)
        self.assertGreater(float(s(out)),float(s(p)))
        torch.testing.assert_close(out['bonds'],out['bonds'].transpose(0,1),atol=0,rtol=0)
        torch.testing.assert_close(out['atomics'][1],p['atomics'][1],atol=0,rtol=0)
        torch.testing.assert_close(out['bonds'][0,0],p['bonds'][0,0],atol=0,rtol=0)

    def test_zero_strength_rng_and_batch_isolation(self):
        p,m,s=self.fixture();rng=torch.get_rng_state().clone()
        out,log=guided_probabilities(p,s,m,strength=0)
        for k in p:torch.testing.assert_close(p[k],out[k],atol=0,rtol=0)
        out,_=guided_probabilities(p,s,m)
        pred={k:torch.stack([v,v,v]) for k,v in p.items()};pred['coords']=torch.zeros(3,2,3)
        guided,cond=apply_to_prediction(pred,pred,2,out)
        for k in p:
            torch.testing.assert_close(guided[k][:2],pred[k][:2],atol=0,rtol=0)
            torch.testing.assert_close(cond[k][2],out[k],atol=0,rtol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(pred['coords'],guided['coords']))

    def test_research_cannot_auto_activate_unmapped_claim(self):
        c=dict(hypothesis='Example',target='T',baseline='native',fragment_mapping={'attachment_validated':False},
            sources=[dict(url='https://example.org/paper',title='Fixture',evidence_type='structure',support='fixture',target_match=True)],
            claimed_endpoint='affinity',transfer_limits=['unknown'],counterevidence=[],expected_tradeoffs=['unknown'],proposed_control={})
        result=assess_hypothesis(c,dict(variable_atom_count=False,categorical_probability_guidance=True))
        self.assertEqual(result['status'],'deferred');self.assertFalse(result['automatic_reward_activation'])
        c['fragment_mapping']['attachment_validated']=True
        c['differentiable_surrogate']='declared coordinate surrogate'
        c['proposed_control']['variables']=['coords']
        result=assess_hypothesis(c,dict(coordinate_guidance=True,categorical_probability_guidance=False))
        self.assertEqual(result['status'],'exploratory_transfer_only')
        c['sources'][0]['transformation_match']=True
        self.assertEqual(assess_hypothesis(c,dict(coordinate_guidance=True))['status'],'candidate_for_validation')


if __name__=='__main__':unittest.main()
