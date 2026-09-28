import copy
import tempfile
import unittest
from pathlib import Path
import torch
from molsteer.molthinker.knowledge import KnowledgeBase
from molsteer.molexecutor.energies import energy, observable
from molsteer.molexecutor.offline import execute_live
from molsteer.molmonitor.checks import gradient_check
from molsteer.molreader.objective_context import compute

ROOT=Path(__file__).resolve().parents[1]
KB=ROOT/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md'


class ThinkerTests(unittest.TestCase):
    def setUp(self):
        self.term=dict(family='flat_bottom_distance',atom_ids=[3,9],lower=1.5,upper=2.,scale=1.,weight=1.)
        self.x=torch.tensor([[0.,0.,0.],[1.,0.,0.]],dtype=torch.float64)

    def test_knowledge_preserves_norms_and_21_rows(self):
        kb=KnowledgeBase(KB)
        self.assertEqual(len(kb.entries),21)
        g=next(e for e in kb.entries if e['function_id']=='G01')
        self.assertIn(r'\|x_i-x_j\|',g['formula'])
        self.assertIn(g['original_row'],KB.read_text(encoding='utf-8'))

    def test_retrieval_and_strategy_distinction(self):
        kb=KnowledgeBase(KB)
        self.assertEqual(kb.retrieve(['local_geometry'])[0]['function_id'],'G01')
        self.assertEqual(next(e for e in kb.entries if e['function_id']=='S04')['role'],'gradient_estimator')
        self.assertEqual(next(e for e in kb.entries if e['function_id']=='S02')['role'],'population_weight')

    def test_unknown_knowledge_row_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kb.md';p.write_text(KB.read_text(encoding='utf-8')+'\n| alien reward | a | b | c | d | e |\n',encoding='utf-8')
            with self.assertRaises(ValueError):KnowledgeBase(p)

    def test_zero_inside_interval_and_correct_descent_direction(self):
        x=self.x.clone().requires_grad_(True)
        self.assertAlmostEqual(float(energy([self.term],x,[3,9])),.125)
        g=torch.autograd.grad(energy([self.term],x,[3,9]),x)[0]
        self.assertLess(float(energy([self.term],x-.1*g,[3,9])),.125)
        x=self.x.clone();x[1,0]=1.7
        self.assertEqual(float(energy([self.term],x,[3,9])),0.)

    def test_gradient_pair_and_translation_invariance(self):
        fn=lambda x:energy([self.term],x,[3,9])
        self.assertTrue(gradient_check(fn,self.x)['passed'])
        self.assertAlmostEqual(float(fn(self.x+100)),float(fn(self.x)))

    def test_minimum_distance_gradient(self):
        t=dict(self.term,family='minimum_distance',atom_ids=[3],reference_coords=[-.5,0.,0.],upper=None)
        self.assertTrue(gradient_check(lambda x:energy([t],x,[3,9]),self.x)['passed'])

    def test_angle_gradient_and_degenerate_rejection(self):
        t=dict(self.term,family='flat_bottom_angle',atom_ids=[3,5,9],lower=100.,upper=130.,scale=10.)
        x=torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
        self.assertTrue(gradient_check(lambda z:energy([t],z,[3,5,9]),x)['passed'])
        x[0]=x[1]
        with self.assertRaises(ValueError):observable(t,x,{3:0,5:1,9:2})

    def test_undeclared_targets_remain_unknown(self):
        self.assertIsNone(compute()['spec']['value'])
        with self.assertRaises(ValueError):compute({'qed':.8})

    def test_live_execution_does_not_silently_use_offline_gradient(self):
        with self.assertRaises(NotImplementedError):execute_live()

    def test_coincident_pair_rejected(self):
        with self.assertRaises(ValueError):energy([self.term],torch.zeros_like(self.x),[3,9])


if __name__=='__main__':unittest.main(verbosity=2)
