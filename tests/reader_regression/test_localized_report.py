import copy
import unittest
import numpy as np
import test_molreader as legacy_fixture
from molreader.packet import build_packet,compute_metric,persistence_signature
from molreader.localized_report import make_localized_report,validate_localized_report


class LocalizedReportTests(unittest.TestCase):
    setUp=legacy_fixture.MolReaderTests.setUp
    save=legacy_fixture.MolReaderTests.save
    ctx=legacy_fixture.MolReaderTests.ctx

    def packet(self):
        return build_packet([self.ctx(v) for v in ('state','prediction','sdf')],
            ['chemistry_context','decode_consistency','valence','protein_clashes','prolif','vina_score'])

    def test_verified_sdf_sources_merge_and_raw_is_separate(self):
        p=self.packet();r=make_localized_report(p)
        self.assertTrue(r['representation_merging']['prediction_sdf_merged'])
        self.assertEqual(len(r['findings']),1)
        self.assertEqual(set(r['findings'][0]['views']),{'prediction','sdf'})
        self.assertTrue(r['raw_state_findings'])
        self.assertTrue(validate_localized_report(r,p))
        self.assertEqual(len(r['evidence_gaps']),2)

    def test_unverified_sdf_is_not_merged(self):
        p=self.packet()
        m=next(m for m in p['observations'] if m['metric_id']=='decode_consistency' and m['view']=='prediction')
        m['values']['difference_count']=1
        r=make_localized_report(p)
        self.assertFalse(r['representation_merging']['prediction_sdf_merged'])
        self.assertEqual(len(r['findings']),2)

    def test_missing_and_altered_evidence_rejected(self):
        p=self.packet();r=make_localized_report(p)
        bad=copy.deepcopy(r);bad['findings'][0]['evidence_ids'].pop()
        with self.assertRaises(ValueError):validate_localized_report(bad,p)
        bad=copy.deepcopy(r);eid=next(iter(bad['evidence_index']))
        bad['evidence_index'][eid]['evidence']['distance_angstrom']=999
        with self.assertRaises(ValueError):validate_localized_report(bad,p)
        bad=copy.deepcopy(r);bad['findings'][0]['repair']='change a bond'
        with self.assertRaises(ValueError):validate_localized_report(bad,p)

    def test_identity_changes_invalidate_persistence(self):
        p=self.packet();m=next(m for m in p['observations'] if m['metric_id']=='protein_clashes' and m['view']=='prediction');e=m['evidence'][0]
        before=persistence_signature(p,m,e)
        q=copy.deepcopy(p)
        c=next(x for x in q['observations'] if x['metric_id']=='chemistry_context' and x['view']=='prediction')
        c['values']['atoms'][0]['element']='N'
        self.assertNotEqual(before,persistence_signature(q,m,e))
        altered=copy.deepcopy(e);altered['catalog']='BRENK';altered['alert']='different_rule'
        self.assertNotEqual(before,persistence_signature(p,m,altered))

    def test_zero_evaluable_pairs_is_not_a_pass(self):
        m=compute_metric(self.ctx(),'intramolecular_clashes')
        self.assertEqual(m['values']['checked_pair_count'],0)
        self.assertIsNone(m['values']['clash_count'])
        self.assertEqual(m['status'],'partial')

    def test_local_valence_survives_unrelated_unknown_atom(self):
        b=copy.deepcopy(self.bundle)
        b['atomics'][0,4]=0;b['atomics'][0,4,0]=1
        for i,j in [(0,2),(2,4)]:
            for a,c in [(i,j),(j,i)]:b['bonds'][0,a,c]=0;b['bonds'][0,a,c,3]=1
        self.save(state=b)
        ctx=self.ctx('state');m=compute_metric(ctx,'valence')
        self.assertTrue(any(e['atom_ids']==[2] and e.get('conservative_valence_lower_bound')==6 for e in m['evidence']))
        angles=compute_metric(ctx,'bond_angles')
        self.assertIsNone(angles['values']['outlier_count'])
        self.assertEqual(angles['values']['evaluated_angle_count'],0)

    def test_optional_gaps_do_not_erase_completed_core_assessment(self):
        p=build_packet([self.ctx()],['valence','connectivity','prolif','vina_score'])
        self.assertEqual(make_localized_report(p)['assessment'],'no_flagged_risks_in_evaluated_metrics')
        p=build_packet([self.ctx()],['prolif','vina_score'])
        self.assertEqual(make_localized_report(p)['assessment'],'insufficient_evidence')

    def test_mmff_local_metric_preserves_coordinates(self):
        ctx=self.ctx();before=ctx.coords.copy();conf=ctx.mol.GetConformer().GetPositions().copy()
        m=compute_metric(ctx,'mmff_local_geometry')
        self.assertEqual(m['status'],'ok')
        np.testing.assert_array_equal(before,ctx.coords)
        np.testing.assert_array_equal(conf,ctx.mol.GetConformer().GetPositions())
        self.assertTrue(all(set(e['atom_ids'])<=set(self.ids) for e in m['evidence']))

    def test_overlapping_catalog_rules_keep_full_region_without_duplicates(self):
        p=build_packet([self.ctx(v) for v in ('prediction','sdf')],['chemistry_context','decode_consistency','structural_alerts'])
        for n,m in enumerate(p['observations']):
            if m['metric_id']!='structural_alerts':continue
            m['evidence']=[dict(evidence_id=f'ev_{n*10+k:020d}',atom_ids=ids,severity='warning',message='Structural screening alert',catalog='test',alert=f'rule{k}') for k,ids in enumerate(([0,2],[2,4]))]
        r=make_localized_report(p);self.assertEqual(len(r['findings']),1)
        c=r['findings'][0]
        self.assertEqual(c['atom_ids'],[0,2,4])
        self.assertEqual(len(c['primary_evidence_ids']),4)
        self.assertEqual(len(c['representative_evidence_ids']),2)
        self.assertEqual(c['supporting_evidence_ids'],[])
        self.assertTrue(validate_localized_report(r,p))


if __name__=='__main__':unittest.main(verbosity=2)
