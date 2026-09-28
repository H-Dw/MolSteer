"""Behavioral tests: provenance, atom identity, missingness and scientific interpretation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from rdkit import Chem
from molreader.io import load_stage,load_config
from molreader.packet import compute_metric,build_packet,validate_packet
from molreader.diagnostic import make_report,validate_report
from molreader.core import write_json,Unavailable


class MolReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage=Path(self.temp.name)/'target_A'/'ligand_000'/'t_0.25'
        self.stage.mkdir(parents=True)
        n=5;self.ids=[0,2,4]
        coords=torch.zeros((1,n,3));coords[0,self.ids]=torch.tensor([[0.,0.,0.],[1.5,0.,0.],[2.0,1.3,0.]])
        atomics=torch.zeros((1,n,15));atomics[:,:,0]=1
        for i,k in zip(self.ids,[3,3,5]):atomics[0,i,0]=0;atomics[0,i,k]=1
        charges=torch.zeros((1,n,8));charges[:,:,1]=1
        bonds=torch.zeros((1,n,n,5));bonds[:,:,:,0]=1
        for i,j in [(0,2),(2,4)]:
            for a,b in [(i,j),(j,i)]:bonds[0,a,b,0]=0;bonds[0,a,b,1]=1
        self.bundle=dict(coords=coords,atomics=atomics,charges=charges,bonds=bonds,mask=torch.tensor([[1,0,1,0,1]]))
        self.bundle['coords'][0,1]=float('nan') # Padding must never leak into measurements.
        self.bundle['affinity']={k:torch.tensor([[v]]) for k,v in dict(pic50=4.1,pki=5.2,pkd=3.3,pec50=6.4).items()}
        self.save()
        mol=Chem.MolFromSmiles('CCO');conf=Chem.Conformer(3)
        for i,p in enumerate([[10.,20.,30.],[11.5,20.,30.],[12.,21.3,30.]]):conf.SetAtomPosition(i,p)
        mol.AddConformer(conf);Chem.MolToMolFile(mol,str(self.stage/'ligand.sdf'))
        self.receptor=Path(self.temp.name)/'protein.pdb'
        self.receptor.write_text('ATOM      1  CA  ALA A   1      10.500  20.000  30.000  1.00 20.00           C  \nEND\n')

    def save(self,state=None,head=None,world=None):
        head=copy.deepcopy(head if head is not None else self.bundle)
        state=copy.deepcopy(state if state is not None else self.bundle)
        if world is None:
            world=copy.deepcopy(head);world['coords']=world['coords']+torch.tensor([10.,20.,30.])
        for name,data in [('state',state),('structure_affinity_prediction',head),('world_prediction',world)]:torch.save(data,self.stage/(name+'.pt'))
        write_json(self.stage/'predictions.json',dict(stage_t=0.25,affinity={k:float(v.item()) for k,v in head['affinity'].items()}))

    def ctx(self,view='prediction',**kwargs):return load_stage(self.stage,view,receptor=kwargs.pop('receptor',self.receptor),**kwargs)

    def test_noncontiguous_mask_and_padding(self):
        c=self.ctx();self.assertEqual(c.atom_ids.tolist(),self.ids);self.assertTrue(c.world_valid);self.assertTrue(c.sdf_mapping_valid)
        m=compute_metric(c,'tensor_integrity');self.assertEqual(m['evidence'],[])
        self.assertEqual(compute_metric(c,'molecular_weight')['status'],'ok')

    def test_localization_preserves_tensor_slots(self):
        m=compute_metric(self.ctx(),'protein_clashes')
        self.assertGreater(m['values']['clash_count'],0)
        self.assertTrue(all(set(e['atom_ids'])<=set(self.ids) for e in m['evidence']))
        self.assertEqual(m['evidence'][0]['residue_id'],'A:1:ALA')

    def test_world_corruption_blocks_interface(self):
        w=copy.deepcopy(self.bundle);w['coords']=w['coords']+torch.tensor([10.,20.,30.]);w['coords'][0,2,0]+=5
        self.save(world=w);c=self.ctx();self.assertFalse(c.world_valid)
        self.assertEqual(compute_metric(c,'protein_clashes')['status'],'unavailable')
        self.assertTrue(compute_metric(c,'tensor_integrity')['evidence'])

    def test_active_nan_is_not_a_clean_result(self):
        b=copy.deepcopy(self.bundle);b['atomics'][0,0,0]=float('nan');self.save(head=b)
        c=self.ctx();self.assertTrue(compute_metric(c,'tensor_integrity')['evidence'])
        self.assertEqual(compute_metric(c,'formal_charge')['status'],'unavailable')

    def test_logits_are_not_silently_softmaxed(self):
        b=copy.deepcopy(self.bundle);b['atomics'][0,0,3]=9;self.save(head=b)
        self.assertEqual(compute_metric(self.ctx(),'atom_confidence')['status'],'unavailable')

    def test_formal_charge_pad_is_unknown(self):
        b=copy.deepcopy(self.bundle);b['charges'][0,0]=0;b['charges'][0,0,0]=1;self.save(state=b)
        c=self.ctx('state');m=compute_metric(c,'formal_charge')
        self.assertIsNone(m['values']['net_formal_charge'])
        self.assertEqual(compute_metric(c,'logp')['status'],'unavailable')
        self.assertEqual(compute_metric(c,'charge_confidence')['values']['one_hot_fraction'],1.)

    def test_no_sdf_mapping_assumption(self):
        mol=Chem.MolFromMolFile(str(self.stage/'ligand.sdf'),removeHs=False)
        mol.GetConformer().SetAtomPosition(0,[20.,20.,20.]);Chem.MolToMolFile(mol,str(self.stage/'ligand.sdf'))
        self.assertFalse(self.ctx().sdf_mapping_valid)
        with self.assertRaises(Unavailable):self.ctx('sdf')

    def test_self_bonds_block_derived_chemistry(self):
        b=copy.deepcopy(self.bundle);b['bonds'][0,0,0,0]=0;b['bonds'][0,0,0,1]=1;self.save(head=b)
        c=self.ctx();self.assertTrue(compute_metric(c,'tensor_integrity')['evidence'])
        self.assertEqual(compute_metric(c,'connectivity')['status'],'unavailable')

    def test_missing_receptor_not_zero_clashes(self):
        c=self.ctx(receptor=Path(self.temp.name)/'missing.pdb')
        m=compute_metric(c,'protein_clashes');self.assertEqual(m['status'],'unavailable');self.assertNotIn('clash_count',m['values'])

    def test_affinity_endpoints_preserved(self):
        m=compute_metric(self.ctx(),'affinity')
        self.assertEqual(set(m['values']),{'pic50','pki','pkd','pec50'})
        self.assertAlmostEqual(m['values']['pki'],5.2,places=5)

    def test_mmff_does_not_mutate_pose(self):
        c=self.ctx();before=c.coords.copy();conf=c.mol.GetConformer().GetPositions().copy()
        m=compute_metric(c,'mmff_strain');self.assertIn(m['status'],['ok','partial'])
        np.testing.assert_array_equal(c.coords,before);np.testing.assert_array_equal(c.mol.GetConformer().GetPositions(),conf)
        self.assertGreaterEqual(m['values']['strain_proxy_kcal_mol'],-0.001)

    def test_external_metrics_reject_cross_molecule(self):
        c=self.ctx();m=compute_metric(c,'molecular_weight');m['identity']['ligand_id']='ligand_999'
        path=Path(self.temp.name)/'external.json';write_json(path,m)
        with self.assertRaises(ValueError):build_packet([self.ctx()],['shape'],[path])

    def test_external_metrics_valid_merge_and_duplicate_rejection(self):
        c=self.ctx();path=Path(self.temp.name)/'external.json';write_json(path,compute_metric(c,'molecular_weight'))
        packet=build_packet([c],['shape'],[path]);self.assertEqual(len(packet['observations']),2)
        with self.assertRaises(ValueError):build_packet([c],['molecular_weight'],[path])

    def test_evidence_binding_and_risk_only_contract(self):
        packet=build_packet([self.ctx()],['protein_clashes','prolif']);report=make_report(packet)
        self.assertEqual(report['assessment'],'risks_observed');self.assertTrue(validate_report(report,packet))
        changed=copy.deepcopy(report);changed['findings'][0]['localized_examples'][0]['distance_angstrom']=999
        with self.assertRaises(ValueError):validate_report(changed,packet)
        changed=copy.deepcopy(report);changed['recommendations']=['Optimize pose']
        with self.assertRaises(ValueError):validate_report(changed,packet)
        changed=copy.deepcopy(packet);changed['observations'][0]['evidence'][0]['atom_ids']=[999]
        with self.assertRaises(ValueError):validate_packet(changed)

    def test_empty_risk_with_missing_evidence_is_not_healthy(self):
        report=make_report(build_packet([self.ctx()],['prolif','vina_score']))
        self.assertEqual(report['assessment'],'insufficient_evidence')

    def test_future_history_is_rejected(self):
        c=self.ctx();packet=build_packet([c],['shape'])
        with self.assertRaises(ValueError):build_packet([c],['shape'],history_packet=packet)

    def test_adapter_shape_mismatch_rejected(self):
        config=load_config();config['atomic_tokens']=config['atomic_tokens'][:-1]
        with self.assertRaises(ValueError):self.ctx(config=config)

if __name__=='__main__':unittest.main(verbosity=2)
