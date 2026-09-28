import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from molreader.io import load_config
from molsteer.molexecutor.mmff_bridge import MMFFStrain
from molsteer.molexecutor.interface_energy import measure
from molsteer.molexecutor.chemistry import encode_mol,decode_endpoint,signature
from molsteer.molexecutor.graph_search import propose,inject_hypothesis
from molsteer.molexecutor.outcome_reward import OutcomeAwareReward,OutcomeObservables,conflict_aware_gradient
from molsteer.molexecutor.outcome_engine import oracle_failures
from molsteer.molthinker.outcomes import revise_from_evidence


def molecule(smiles='CCO'):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles));AllChem.EmbedMolecule(m,randomSeed=13);AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def endpoint(mol):
    vocab=load_config();n=mol.GetNumAtoms()
    p=dict(coords=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.double),
        atomics=torch.zeros(n,len(vocab['atomic_tokens']),dtype=torch.double),charges=torch.zeros(n,len(vocab['charge_tokens']),dtype=torch.double),
        bonds=torch.zeros(n,n,len(vocab['bond_orders']),dtype=torch.double),affinity=dict(pkd=torch.tensor(5.,dtype=torch.double)))
    return encode_mol(mol,vocab,p)


def protein():
    return dict(donors=[dict(coords=[0.,3.,0.],hydrogen=[0.,2.,0.])],acceptors=[dict(coords=[3.,0.,0.],direction=[-1.,0.,0.])],
        heavy_coords=[[0.,3.,0.],[3.,0.,0.]],heavy_radii=[1.55,1.52])


class OutcomeGuidanceTests(unittest.TestCase):
    def test_mmff_envelope_derivative_and_fixed_input(self):
        mol=molecule();x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.double,requires_grad=True)
        with torch.no_grad():x[2,0]+=.08
        before=mol.GetConformer().GetPositions().copy();oracle=MMFFStrain();value=oracle.tensor(x,mol)
        g=torch.autograd.grad(value,x)[0];d=torch.zeros_like(x);d[2,0]=1e-4
        fd=(oracle.tensor(x+d,mol)-oracle.tensor(x-d,mol))/(2e-4)
        self.assertAlmostEqual(float(g[2,0]),float(fd),delta=.02)
        self.assertTrue(np.array_equal(before,mol.GetConformer().GetPositions()))
        self.assertLess(float(oracle.tensor(x-.0001*g,mol)),float(value))

    def test_interface_proxy_derivative_is_finite(self):
        mol=molecule();x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.double,requires_grad=True)
        p=protein();v=measure(x,mol,p);loss=v['contact']-v['desolvation'];g=torch.autograd.grad(loss,x)[0]
        d=torch.zeros_like(x);d[2,1]=1e-5
        a,b=measure(x+d,mol,p),measure(x-d,mol,p)
        fd=((a['contact']-a['desolvation'])-(b['contact']-b['desolvation']))/(2e-5)
        self.assertAlmostEqual(float(g[2,1]),float(fd),places=5)
        self.assertTrue(torch.isfinite(g).all())

    def test_protein_donor_direction_changes_contact(self):
        mol=molecule('CC=O');x=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.double)
        oxygen=x[2].numpy();p=protein();p['acceptors']=[]
        p['donors']=[dict(coords=(oxygen+[0,2.9,0]).tolist(),hydrogen=(oxygen+[0,1.9,0]).tolist())]
        good=measure(x,mol,p)['contact'];p['donors'][0]['hydrogen']=(oxygen+[0,3.9,0]).tolist()
        bad=measure(x,mol,p)['contact'];self.assertGreater(float(good),float(bad)+.5)

    def test_natural_improvement_is_not_attributed_to_guidance(self):
        mol=molecule();p=endpoint(mol);obs=OutcomeObservables(protein())
        x=OutcomeAwareReward.__new__(OutcomeAwareReward)
        x.observables=obs;x.vocab=load_config();x.receptor=[dict(coords=[0,3,0])]
        x.spec=dict(affinity_head='pkd',pocket_contact_distance=4.5,pocket_distance_scale=1.,
            outcome_weights=dict(affinity=1.,strain=1.,contact=.5,desolvation=.25,pocket=1.),
            outcome_scales=dict(strain_kcal_mol=5.,contact=2.,desolvation=2.))
        frames=[]
        for t,offset in [(.5,.1),(1.,0.)]:
            point={**p,'coords':p['coords'].clone(),'affinity':dict(pkd=torch.tensor(5.+t,dtype=torch.double))}
            point['coords'][2,0]+=offset
            v,_,_=obs.evaluate(point,x.vocab)
            frames.append(dict(time=t,status='ok',observables={k:float(z) for k,z in v.items()},affinity=dict(pkd=5.+t),pocket=0.))
            x.reference=dict(frames=frames);x.time=t
            value,_=x.evaluate(point);self.assertAlmostEqual(float(value),0.,places=8)

    def test_gradient_projection_protects_structure(self):
        x=torch.tensor([1.,2.],requires_grad=True)
        parts=dict(strain=x[0],affinity=-3*x[0]+x[1],contact=0*x.sum(),desolvation=0*x.sum(),pocket=0*x.sum())
        g,detail=conflict_aware_gradient(parts,x)
        self.assertGreaterEqual(float(g[0]),1.)
        self.assertTrue(detail['components']['affinity']['conflicted_with_strain'])
        self.assertFalse(detail['direction_is_raw_reward_gradient'])

    def test_discrete_trials_are_real_and_do_not_touch_other_batch_or_rng(self):
        p=endpoint(molecule());vocab=load_config();rng=torch.get_rng_state().clone()
        candidates,info=propose(p,vocab,[0,1,2],max_candidates=8)
        self.assertTrue(candidates);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        state={k:torch.stack([v,v,v]) for k,v in p.items() if torch.is_tensor(v)}
        before={k:v.clone() for k,v in state.items()};new=inject_hypothesis(state,2,candidates[0])
        for key in state:torch.testing.assert_close(state[key],before[key],atol=0,rtol=0)
        for key in state:torch.testing.assert_close(new[key][:2],state[key][:2],atol=0,rtol=0)
        torch.testing.assert_close(new['coords'],state['coords'],atol=0,rtol=0)
        self.assertTrue(any(not torch.equal(new[k],state[k]) for k in ('atomics','charges','bonds')))
        for c in candidates:self.assertIsNotNone(decode_endpoint(c['endpoint'],vocab))

    def test_aromatic_and_tautomer_hypotheses_round_trip(self):
        mol=molecule('Oc1ncccc1');p=endpoint(mol)
        self.assertEqual(Chem.MolToSmiles(decode_endpoint(p,load_config())),Chem.MolToSmiles(mol))
        candidates,info=propose(p,load_config(),list(range(mol.GetNumAtoms())),max_candidates=16,max_changed_slots=7)
        self.assertGreater(info['valid_unique_hypotheses'],2)
        self.assertGreater(info['family_counts'].get('tautomer',0),0)

    def test_counterevidence_revises_objectives_not_eta(self):
        p=dict(program_id='parent',feedback_policy=dict(conflict_streak=3,affinity_gain_trigger=.03,
            score_worsening_kcal_mol=.15,strain_worsening_kcal_mol=2.),outcome_weights=dict(affinity=1.,strain=1.,contact=.5,desolvation=.25,pocket=1.))
        before=copy.deepcopy(p)
        self.assertIsNone(revise_from_evidence(p,dict(affinity_delta=.2,vina_delta=-.1,strain_delta=-3.)))
        child=revise_from_evidence(p,dict(affinity_delta=.2,vina_delta=.3,strain_delta=0.))
        self.assertEqual(p,before);self.assertLess(child['outcome_weights']['affinity'],1.)
        self.assertFalse(child['revision']['eta_changed']);self.assertNotEqual(child['program_id'],'parent')

    def test_aromatic_representation_is_not_a_new_chemical_identity(self):
        mol=molecule('c1ccccc1');p=endpoint(mol);q=dict(p);q['bonds']=p['bonds'].clone()
        kek=Chem.Mol(mol);Chem.Kekulize(kek,clearAromaticFlags=True);vocab=load_config()
        for b in kek.GetBonds():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();idx=vocab['bond_orders'].index(b.GetBondTypeAsDouble())
            q['bonds'][i,j].zero_();q['bonds'][j,i].zero_();q['bonds'][i,j,idx]=1;q['bonds'][j,i,idx]=1
        self.assertFalse(torch.equal(p['bonds'],q['bonds']))
        self.assertEqual(signature(decode_endpoint(p,vocab)),signature(decode_endpoint(q,vocab)))
        trials,_=propose(p,vocab,list(range(6)))
        for trial in trials:self.assertNotEqual(signature(trial['molecule']),signature(mol))

    def test_external_regression_cannot_be_erased_by_head_gain(self):
        reasons=oracle_failures(dict(vina=-4.,strain=25.,pkd=100.),dict(vina=-6.,strain=15.),
            dict(vina_regression_allowance=.15,strain_regression_allowance=2.))
        self.assertEqual(set(reasons),{'independent_vina_regression','independent_self_strain_regression'})


if __name__=='__main__':unittest.main()
