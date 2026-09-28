import copy
import json
import unittest
from pathlib import Path
import torch
from molsteer.molexecutor.categorical import temper_prediction
from molsteer.molexecutor.affinity import AffinityStructureReward
from molsteer.molthinker.affinity import build_affinity_program
from molsteer.molthinker.creativity import build_program


class AffinityTests(unittest.TestCase):
    def test_temperature_conserves_other_batch_and_self_condition(self):
        p=torch.tensor([[[.6,.3,.1],[.97,.02,.01]]]*3)
        bonds=torch.tensor([[[[1.,0.],[.6,.4]],[[.6,.4],[1.,0.]]]]*3)
        pred=dict(atomics=p,charges=p,bonds=bonds,coords=torch.ones(3,2,3))
        cond=dict(pred);state=torch.get_rng_state().clone()
        q,c=temper_prediction(pred,cond,2,2.)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for key in ['atomics','charges','bonds']:
            torch.testing.assert_close(q[key][:2],pred[key][:2],atol=0,rtol=0)
            torch.testing.assert_close(q[key],c[key],atol=0,rtol=0)
            torch.testing.assert_close(q[key].sum(-1),torch.ones_like(q[key].sum(-1)))
        self.assertLess(float(q['atomics'][2,0,0]),.6)
        torch.testing.assert_close(q['atomics'][2,1],p[2,1],atol=0,rtol=0)
        torch.testing.assert_close(q['bonds'][2],q['bonds'][2].transpose(0,1),atol=0,rtol=0)
        self.assertIs(temper_prediction(pred,cond,2,1.)[0],pred)

    def test_same_time_control_and_live_gradient(self):
        reward=AffinityStructureReward.__new__(AffinityStructureReward)
        reward.spec=dict(affinity_head='pkd',affinity_scale=1.,affinity_saturation=2.)
        reward.control=[(.5,5.),(1.,6.)];reward.control_times=[.5,1.];reward.set_time(.75)
        x=torch.tensor(6.,dtype=torch.float64,requires_grad=True)
        utility,value=reward.affinity_utility(dict(affinity=dict(pkd=x)))
        self.assertAlmostEqual(reward.control_value(),5.5)
        derivative=torch.autograd.grad(utility,x)[0]
        self.assertGreater(float(derivative),0.)
        self.assertLess(float(derivative),1.)
        with self.assertRaisesRegex(ValueError,'Missing finite live affinity'):
            reward.affinity_utility({})
        reward.set_time(.1)
        with self.assertRaisesRegex(ValueError,'does not cover'):reward.control_value()

    def test_intent_is_explicit_and_source_unchanged(self):
        path=Path(__file__).resolve().parents[1]/'examples/5i0b_A__5vef_M77/ligand_002/t_0.50/RewardSpec.json'
        old=build_program(json.loads(path.read_text()));snapshot=copy.deepcopy(old)
        with self.assertRaises(ValueError):build_affinity_program(old,{})
        new=build_affinity_program(old,dict(primary_affinity='pkd',direction='maximize'))
        self.assertEqual(old,snapshot);self.assertEqual(new['lambda_graph'],0.)
        self.assertEqual(new['packet_id'],old['packet_id'])
        self.assertFalse(new['control_contract']['atom_count_change'])

    def test_large_affinity_cannot_compensate_a_new_severe_clash(self):
        from molreader.io import load_config
        vocab=load_config()
        path=Path(__file__).resolve().parents[1]/'examples/5i0b_A__5vef_M77/ligand_002/t_0.50/RewardSpec.json'
        source=build_program(json.loads(path.read_text()));source['region_atom_ids']=[0,1]
        program=build_affinity_program(source,dict(primary_affinity='pkd',direction='maximize'))
        def onehot(indices,classes):return torch.nn.functional.one_hot(torch.tensor(indices),classes).double()
        baseline=dict(coords=torch.tensor([[0.,0.,0.],[1.43,0.,0.]],dtype=torch.float64),
            atomics=onehot([vocab['atomic_tokens'].index('C'),vocab['atomic_tokens'].index('O')],len(vocab['atomic_tokens'])),
            charges=onehot([vocab['charge_tokens'].index(0)]*2,len(vocab['charge_tokens'])),
            bonds=onehot([[0,1],[1,0]],len(vocab['bond_orders'])))
        receptor=[dict(coords=[0.,4.,0.],vdw_radius=1.7)]
        reward=AffinityStructureReward(program,baseline,receptor,vocab,{'0.5':5.,'1.0':5.})
        candidate=dict(baseline,coords=baseline['coords']+torch.tensor([0.,4.,0.]),affinity=dict(pkd=torch.tensor(100.)))
        self.assertIn('new_or_worsened_severe_clash',reward.feasible(candidate,baseline))
        bad=dict(baseline,bonds=onehot([[0,0],[0,0]],len(vocab['bond_orders'])))
        self.assertTrue(any('Disconnected' in x for x in reward.feasible(bad,baseline)))


if __name__=='__main__':unittest.main()
