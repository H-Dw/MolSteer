import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from molsteer.molreader.pose_comparison import compare
from molsteer.molreader.forcefield_evidence import measure,components
from molsteer.molreader.polar_context import measure as polar_context


class OutcomeEvidenceTests(unittest.TestCase):
    def molecule(self):
        m=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(m,randomSeed=7);return Chem.RemoveHs(m)

    def test_translation_is_not_internal_deformation(self):
        a=self.molecule();b=Chem.Mol(a);conf=b.GetConformer()
        for i in range(b.GetNumAtoms()):conf.SetAtomPosition(i,tuple(np.array(conf.GetAtomPosition(i))+[1,2,3]))
        result=compare(a,b)
        self.assertAlmostEqual(result['world_rmsd_angstrom'],np.sqrt(14))
        self.assertLess(result['aligned_rmsd_angstrom'],1e-8)
        self.assertTrue(result['same_slot_graph'])

    def test_forcefield_preserves_input_and_conserves_components(self):
        mol=self.molecule();before=mol.GetConformer().GetPositions().copy()
        result,h,relaxed=measure(mol)
        self.assertTrue(np.array_equal(before,mol.GetConformer().GetPositions()))
        self.assertTrue(np.array_equal(before,h.GetConformer().GetPositions()[:mol.GetNumAtoms()]))
        self.assertAlmostEqual(sum(result['component_strain'].values()),result['strain_kcal_mol'],places=6)

    def test_different_atom_counts_need_explicit_mapping(self):
        a=self.molecule();b=Chem.MolFromSmiles('CC')
        with self.assertRaises(ValueError):compare(a,b)

    def test_missing_burial_or_contacts_are_unknown(self):
        mol=self.molecule()
        self.assertIsNone(polar_context(mol,[],None)[0]['buried_without_detected_polar_contact'])
        sasa=[dict(atom_id=2,isolated_sasa_angstrom2=10.,complex_sasa_angstrom2=1.,buried_sasa_angstrom2=9.)]
        self.assertIsNone(polar_context(mol,None,sasa)[0]['buried_without_detected_polar_contact'])
        self.assertTrue(polar_context(mol,[],sasa)[0]['buried_without_detected_polar_contact'])


if __name__=='__main__':unittest.main()
