import unittest
import torch
import tempfile
from pathlib import Path
from molsteer.molexecutor.interfaces import CallbackAdapter, GuidanceBudget, guided_step, endpoint_gradient
from molsteer.molexecutor.program import balance, interval, angle


class ExecutorTests(unittest.TestCase):
    def test_live_pullback_matches_finite_difference(self):
        a=CallbackAdapter(lambda x:torch.sin(x)*2,lambda dt:dt)
        x=torch.tensor([[.2,.7,-.4]],dtype=torch.float64)
        reward=lambda y:-(y-.3).square().sum()
        _,g,_=endpoint_gradient(a,x,reward)
        fd=torch.zeros_like(x)
        for j in range(3):
            d=torch.zeros_like(x);d[0,j]=1e-6
            fd[0,j]=(reward(a.predict(x+d))-reward(a.predict(x-d)))/2e-6
        torch.testing.assert_close(g,fd,atol=1e-8,rtol=1e-6)

    def test_flow_guidance_masks_and_budget(self):
        x=torch.ones((2,3),dtype=torch.float64)
        a=CallbackAdapter(lambda y:y,lambda dt:dt)
        budget=GuidanceBudget(strength=2,max_step_angstrom=.1,max_path_angstrom=.12)
        path=torch.tensor([.1,0.],dtype=x.dtype)
        y,p,record=guided_step(a,x,lambda z:-z.square().sum(),lambda z:z,.5,
            torch.tensor([True,False]),path,budget,lambda c,b:True)
        self.assertLess(float(y[0].norm()),float(x[0].norm()))
        torch.testing.assert_close(y[1],x[1],atol=0,rtol=0)
        self.assertAlmostEqual(float(p[0]),.12)

    def test_reverse_diffusion_uses_covariance_not_negative_dt(self):
        x=torch.tensor([[1.,0,0]])
        a=CallbackAdapter(lambda z:z,lambda dt:.05)
        y,_,_=guided_step(a,x,lambda z:-z.square().sum(),lambda z:z,-1,
            torch.tensor([True]),torch.zeros(1),GuidanceBudget(max_step_angstrom=1),lambda c,b:True)
        self.assertLess(float(y[0,0]),1)

    def test_rejected_guidance_preserves_native_step(self):
        x=torch.ones((1,3));native=x+.1
        a=CallbackAdapter(lambda z:z,lambda dt:dt)
        y,path,record=guided_step(a,x,lambda z:-z.square().sum(),lambda z:native,.1,
            torch.tensor([True]),torch.zeros(1),GuidanceBudget(),lambda c,b:False)
        torch.testing.assert_close(y,native,atol=0,rtol=0)
        self.assertEqual(float(path.sum()),0)
        self.assertFalse(record['accepted'])

    def test_balance_large_values_stable_and_rewards_improvement(self):
        x=torch.tensor(1000.,requires_grad=True);y=torch.tensor(1.,requires_grad=True)
        loss=balance([x,y],[1,1],.1,.05)
        gx,gy=torch.autograd.grad(loss,(x,y))
        self.assertTrue(torch.isfinite(loss));self.assertGreater(gx,gy);self.assertGreater(gy,0)
        with self.assertRaises(ValueError):balance([],[],.1,.05)

    def test_angle_derivative_and_interval_dead_zone(self):
        x=torch.tensor([[1.,0,0],[0.,0,0],[1.,1,0]],dtype=torch.float64,requires_grad=True)
        self.assertAlmostEqual(float(angle(x,[0,1,2]).detach()),45)
        self.assertTrue(torch.isfinite(torch.autograd.grad(angle(x,[0,1,2]),x)[0]).all())
        self.assertEqual(float(interval(torch.tensor(1.5),1,2,1)),0)

    def test_nested_runtime_snapshot_is_tensor_only(self):
        from molsteer.molexecutor.flowr import tree_map
        class MappingLike:
            def items(self):return {'cond':torch.ones(2),'flag':True}.items()
        state=tree_map(lambda t:t.detach().cpu(),{'sc':MappingLike(),'times':[torch.ones(1)]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runtime.pt';torch.save(state,path)
            restored=torch.load(path,weights_only=True)
        self.assertTrue(restored['sc']['flag'])
        torch.testing.assert_close(restored['sc']['cond'],torch.ones(2))

    def test_default_reasoning_mode_and_explicit_selection(self):
        import json
        from molsteer.molthinker.creativity import build_program,DEFAULT_MODE,DEFAULT_SKILL
        self.assertEqual(DEFAULT_MODE,'creativity')
        self.assertEqual(DEFAULT_SKILL,'molthinker-reward-creativity')
        root=Path(__file__).resolve().parents[1]
        source=json.loads((root/'examples/5i0b_A__5vef_M77/ligand_002/t_0.50/RewardSpec.json').read_text())
        self.assertEqual(build_program(source)['mode'],'creativity')
        self.assertEqual(build_program(source,'selection')['mode'],'selection')
        with self.assertRaises(ValueError):build_program(source,'unknown')

    def test_native_flow_can_erase_an_injection(self):
        from molsteer.molmonitor.guidance_dynamics import linear_flow_retention
        delta=torch.tensor([[.01,0,0]],dtype=torch.float64)
        result=linear_flow_retention(delta,torch.zeros_like(delta),.01,.99)
        self.assertAlmostEqual(result['local_retention_norm_ratio'],0,places=12)
        identity=linear_flow_retention(delta,delta,.01,.99)
        self.assertAlmostEqual(identity['local_retention_norm_ratio'],1,places=12)

    def test_conflict_and_clipping_are_measured_separately(self):
        from molsteer.molmonitor.guidance_dynamics import step_comparison
        native=torch.tensor([[1.,0,0]])
        gradient=torch.tensor([[-1.,0,0]])
        report=step_comparison(native,gradient,gradient*.5,gradient*.02)
        self.assertAlmostEqual(report['cosine_native_gradient'],-1)
        self.assertEqual(report['clipped_atom_count'],1)
        self.assertAlmostEqual(report['bounded_guidance_to_native_ratio'],.02,places=6)

    def test_rng_checkpoint_reproduces_python_numpy_and_torch_draws(self):
        import random
        import numpy as np
        from molsteer.molexecutor.flowr import snapshot_rng,restore_rng
        initial=snapshot_rng()
        def draw():
            return (random.random(),np.random.random(3),torch.rand(3),
                    [torch.rand(3,device=f'cuda:{i}').cpu() for i in range(torch.cuda.device_count())])
        try:
            state=snapshot_rng();first=draw();restore_rng(state);second=draw()
            self.assertEqual(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
            torch.testing.assert_close(first[2],second[2],atol=0,rtol=0)
            for x,y in zip(first[3],second[3]):torch.testing.assert_close(x,y,atol=0,rtol=0)
        finally:restore_rng(initial)

    def test_native_resume_rejects_wrong_cuda_rng_device(self):
        from molsteer.molexecutor.flowr import FlowrRootAdapter
        adapter=FlowrRootAdapter.__new__(FlowrRootAdapter)
        adapter.source_hash='source';adapter.precision='highest';adapter.device=torch.device('cuda:1')
        adapter.config=dict(checkpoint='model',receptor='r',reference_ligand='l',target_id='t')
        checkpoint=dict(format='flowr_root_live_runtime',source_hash='source',model_checkpoint='model',
            config=dict(receptor='r',reference_ligand='l',target_id='t'),precision='highest',cuda_device_index=0)
        with self.assertRaisesRegex(ValueError,'CUDA generator device differs'):adapter.restore(checkpoint)


if __name__=='__main__':unittest.main()
