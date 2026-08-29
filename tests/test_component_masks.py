import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from component_mask_metrics import pole_end_candidate,mask_matches,decode_masks
from resume_component_masks import training_audit
from acquire_uk_material_sources import SOURCES,jpeg_size
from download_mpid import FILES as MPID_FILES
from audit_mpid_archives import origin_key
from sample_mpid_units import choose


class ComponentMaskTests(unittest.TestCase):
    def test_material_protocol_separates_localisation_material_and_abstention(self):
        protocol=json.loads((ROOT/'configs/uk_insulator_material_v1.json').read_text())
        self.assertEqual([c['name'] for c in protocol['classes']],
                         ['glass','porcelain_ceramic','polymer_composite'])
        self.assertEqual(protocol['abstention_label'],'unknown')
        self.assertFalse(protocol['abstention_is_trained_material_class'])
        self.assertFalse(protocol['data_contract']['random_image_split_allowed'])
        self.assertFalse(protocol['data_contract']['pseudo_labels_allowed_as_evaluation_truth'])
        self.assertIn('localisation baseline only',protocol['architecture']['existing_yolo26_supervised_role'])
        self.assertTrue(protocol['uk_final_gate']['freeze_before_inference'])

    def test_uk_material_sources_preserve_groups_and_do_not_claim_truth(self):
        self.assertGreaterEqual(len(SOURCES),9)
        ids=[row['photo_id'] for row in SOURCES]
        self.assertEqual(len(ids),len(set(ids)))
        grouped={row['photo_id']:row['asset_group'] for row in SOURCES}
        self.assertEqual(grouped['3209028'],grouped['3208894'])
        self.assertNotEqual(grouped['3209028'],grouped['3809215'])
        self.assertNotEqual(grouped['770248'],grouped['6714446'])
        self.assertTrue(all(row['evidence'] and row['author'] for row in SOURCES))
        self.assertTrue(all('candidate' in row['use'] or 'legacy' in row['use'] or 'auxiliary' in row['use'] for row in SOURCES))
        audit=json.loads((ROOT/'docs/UK_MATERIAL_SOURCE_PIXEL_AUDIT_V1.json').read_text())
        self.assertFalse(audit['model_inference_performed'])
        self.assertEqual(audit['summary']['records'],len(SOURCES))
        self.assertIn('model output never supplies their truth',audit['next_gate'])

    def test_jpeg_size_rejects_non_jpeg(self):
        with self.assertRaises(ValueError):jpeg_size(b'not an image')

    def test_mpid_download_pins_all_three_archives(self):
        self.assertEqual({name.split('_')[0] for name,_ in MPID_FILES},{'glass','porcelain','composite'})
        self.assertTrue(all(len(md5)==32 for _,md5 in MPID_FILES))
        self.assertEqual(origin_key('glass/train/images/DJI_0123_jpg.rf.abcdef0123.jpg'),'dji_0123')
        rows=[{'sha256':str(i),'feature':{'min_area':i/100,'max_area':i/100,'instances':i+1,'min_aspect':1/(i+1),'max_aspect':i+1}} for i in range(1,10)]
        selected=choose(rows)
        self.assertEqual(len({row['sha256'] for row in selected}),len(selected))

    def test_combined_uk_pool_keeps_provenance_separate_from_truth(self):
        import build_uk_source_pool
        if not (ROOT/'data/external/uk_material_sources_v1/manifest.json').exists():
            self.skipTest('Optional acquired UK material sources are not in the source release')
        build_uk_source_pool.main()
        pool=json.loads((ROOT/'data/external/uk_source_pool_v1/manifest.json').read_text())
        self.assertEqual(pool['count'],36)
        self.assertEqual(pool['counts']['provenance_only'],27)
        self.assertEqual(pool['counts']['source_evidenced_material_candidate'],9)
        self.assertEqual(pool['counts']['new_primary_material_candidates'],5)
        self.assertFalse(pool['split_frozen'])

    def test_duplicate_final_callback_is_not_an_extra_training_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training').mkdir()
            events=[{'epoch':i,'losses':[1.]} for i in range(1,21)]
            (root/'results.json').write_text(json.dumps({'status':'FAILED','predictions':[],'epoch_losses':events+[events[-1]]}))
            (root/'training/results.csv').write_text('epoch,time,train/seg_loss\n'+''.join(f'{i},{i},1\n' for i in range(1,21)))
            _,rows=training_audit(root);self.assertEqual(len(rows),20)
            (root/'training/results.csv').write_text('epoch,time,train/seg_loss\n1,1,1\n')
            with self.assertRaises(AssertionError):training_audit(root)

    def numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('Optional raster checks require NumPy')
        return np

    def test_pole_end_is_unscored_and_abstains_without_context(self):
        np=self.numpy();mask=np.zeros((110,110),bool);mask[10:100,45:55]=True
        self.assertEqual(pole_end_candidate(mask,[])['status'],'unknown')
        r=pole_end_candidate(mask,[[20,12,80,20]])
        self.assertEqual(r['status'],'geometry_candidate');self.assertLess(r['point'][1],20)
        self.assertIsNone(r['score']);self.assertFalse(r['supervised_pole_top'])
        self.assertEqual(pole_end_candidate(mask,[[20,12,80,20],[20,90,80,98]])['status'],'unknown')
        mask=np.zeros((110,110),bool);mask[45:55,10:100]=True
        r=pole_end_candidate(mask,[[12,20,20,80]])
        self.assertLess(r['point'][0],20)

    def test_clipped_pole_end_is_not_accepted(self):
        np=self.numpy();mask=np.zeros((100,100),bool);mask[:,45:55]=True
        self.assertIn('truncated',pole_end_candidate(mask,[[20,0,80,10]])['reason'])

    def test_mask_matching_counts_duplicates_and_missing_classes(self):
        np=self.numpy();mask=np.ones((10,10),bool)
        predictions=[{'prediction_index':i,'class_id':0,'score':.9-i*.1} for i in range(2)]
        refs=[{'class_id':0},{'class_id':1}]
        result=mask_matches(predictions,[mask,mask],refs,[mask,mask])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,1))
        raw={'mask_shape':np.array([0,10,10]),'mask_bits':np.zeros((0,13),np.uint8)}
        self.assertEqual(decode_masks(raw).shape,(0,10,10))


if __name__=='__main__':unittest.main()
