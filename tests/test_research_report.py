import json,unittest
from molsteer.reporting.research_report import embedded_json,mol_svg
from rdkit import Chem


class ResearchReportTests(unittest.TestCase):
    def test_retrieved_text_cannot_break_the_embedded_data_boundary(self):
        value={'paper':'</script><script>alert(1)</script>& \u2028','unknown':None}
        data=embedded_json(value)
        self.assertNotIn('</script>',data);self.assertEqual(json.loads(data),value)

    def test_molecular_diagram_does_not_mutate_saved_coordinates(self):
        mol=Chem.MolFromSmiles('CCO');conf=Chem.Conformer(3)
        for i in range(3):conf.SetAtomPosition(i,(i,.3*i,.7))
        mol.AddConformer(conf);before=mol.GetConformer().GetPositions().copy();svg=mol_svg(mol,[1])
        self.assertIn('<svg',svg);self.assertTrue((before==mol.GetConformer().GetPositions()).all())


if __name__=='__main__':unittest.main()
