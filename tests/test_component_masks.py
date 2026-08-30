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
from prepare_ttpla_steelwork_demo import yolo_lines
from acquire_uk_material_prospective_v1 import SOURCES as PROSPECTIVE_MATERIAL_SOURCES
from roihu_uk_material_prospective_v1 import asset_counts,greedy_matches
from acquire_uk_insulator_localisation_v1 import SOURCES as LOCALISATION_SOURCES
from roihu_uk_insulator_localisation_v1 import axis_starts,fixed_priority_fusion,match_counts


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

    def test_ttpla_protocol_keeps_assembly_scope_and_source_groups(self):
        cfg=json.loads((ROOT/'configs/ttpla_steelwork_demo_v1.json').read_text())
        self.assertIn('assembly-level',cfg['scope_warning'])
        self.assertIn('not an individual steel member',cfg['scope_warning'])
        self.assertFalse(cfg['selection']['group_overlap'])
        self.assertEqual(cfg['model']['checkpoint_selection'],'final epoch; no test or UK selection')
        self.assertIn('No automatic extension',cfg['budget'])

    def test_ttpla_boundary_noise_is_recorded_not_silently_clipped(self):
        row={'row':{'file_name':'sample.jpg','width':100,'height':80,
                    'annotations':{'category_name':['tower_lattice'],
                                   'segmentation':[[0,-1.25,100,0,100,80,0,80]]}}}
        lines,corrections=yolo_lines(row)
        self.assertEqual(len(lines),1)
        self.assertEqual(corrections,[{'coordinate_index':1,'raw':-1.25,'clipped':0.0,'axis':'y'}])
        row['row']['annotations']['segmentation'][0][1]=-2.1
        with self.assertRaises(ValueError):yolo_lines(row)

    def test_pole_top_protocol_is_geometry_target_not_physical_truth(self):
        cfg=json.loads((ROOT/'configs/pole_top_keypoint_v1.json').read_text())
        self.assertIn('publisher masks',cfg['target'])
        self.assertIn('not an annotated physical tip',cfg['target'])
        self.assertEqual(cfg['split'],'dev')
        self.assertIn('No new inference',cfg['model_execution'])
        result_path=ROOT/'runs/pole_top_keypoint/v1_20260829/results.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            self.assertFalse(result['target_is_physical_tip_annotation'])
            self.assertFalse(result['target_is_model_pseudo_label'])
            self.assertTrue(result['target_is_publisher_mask_derived'])
            self.assertEqual(result['summary']['mask_model']['accepted'],10)
            self.assertEqual(result['summary']['box_model']['accepted'],8)

    def test_uk_material_prospective_split_is_asset_disjoint_and_predeclared(self):
        adaptation={r['asset_group'] for r in PROSPECTIVE_MATERIAL_SOURCES if r['role']=='adaptation'}
        test={r['asset_group'] for r in PROSPECTIVE_MATERIAL_SOURCES if r['role']=='prospective_test'}
        self.assertEqual(len(adaptation),3)
        self.assertEqual(len(test),5)
        self.assertFalse(adaptation & test)
        self.assertEqual(sum(len(r['regions']) for r in PROSPECTIVE_MATERIAL_SOURCES if r['role']=='prospective_test'),18)
        self.assertFalse(any(r['material']=='polymer_composite' for r in PROSPECTIVE_MATERIAL_SOURCES))
        self.assertTrue(all(r['exclusion_reason'] for r in PROSPECTIVE_MATERIAL_SOURCES if r['role']=='excluded'))

    def test_prospective_region_matching_is_one_to_one(self):
        refs=[{'xyxy':[0,0,10,10]},{'xyxy':[20,0,30,10]}]
        predictions=[{'box':[0,0,10,10]},{'box':[1,0,11,10]},{'box':[20,0,30,10]}]
        matches=greedy_matches(refs,predictions,.3)
        self.assertEqual([(m['reference_index'],m['prediction_index']) for m in matches],[(1,2),(0,0)])

    def test_asset_decision_requires_consistent_accepted_regions(self):
        rows=[{'record_id':'a','expected_material':'glass','decision':{'material':'glass'}},
              {'record_id':'a','expected_material':'glass','decision':{'material':'unknown'}},
              {'record_id':'b','expected_material':'porcelain_ceramic','decision':{'material':'glass'}},
              {'record_id':'b','expected_material':'porcelain_ceramic','decision':{'material':'porcelain_ceramic'}}]
        result=asset_counts(rows)
        self.assertEqual(result['targets'],2)
        self.assertEqual(result['accepted'],1)
        self.assertEqual(result['correct_accepted'],1)

    def test_prospective_result_and_report_keep_claim_boundaries(self):
        result_path=ROOT/'runs/material_head/v3_uk_prospective_20260830/results.json'
        report_path=ROOT/'runs/uk_capabilities/v3_20260827/report/material_prospective/data.json'
        if not result_path.exists() or not report_path.exists():
            self.skipTest('Optional prospective Roihu output is not in the source release')
        result=json.loads(result_path.read_text())
        report=json.loads(report_path.read_text())
        self.assertEqual(result['status'],'COMPLETE')
        self.assertFalse(result['test_used_for_training_or_selection'])
        self.assertEqual(result['encoder_gradient_steps'],0)
        self.assertEqual(result['head_gradient_steps'],120)
        adapted=result['oracle_diagnostics']['adapted']['regions']
        self.assertEqual((adapted['correct_accepted_material_targets'],adapted['accepted_material_targets']),(12,14))
        self.assertEqual(result['localisation_diagnostics']['matched_regions'],1)
        self.assertFalse(result['localisation_diagnostics']['reference_regions_are_expert_ground_truth'])
        self.assertFalse(result['outputs_are_probabilities'])
        self.assertFalse(result['deployment_claim'])
        self.assertEqual(len(report['gallery']),5)
        self.assertFalse(report['reference_regions_are_expert_ground_truth'])
        self.assertFalse(report['uk_population_accuracy_claim'])

    def test_uk_localisation_acceptance_is_frozen_and_grouped(self):
        accepted=[r for r in LOCALISATION_SOURCES if r['role'] in {'prospective_test','hard_negative'}]
        self.assertEqual(len(accepted),8)
        self.assertEqual(sum(len(r['boxes']) for r in accepted),40)
        self.assertEqual(len({r['asset_group'] for r in accepted}),7)
        self.assertEqual(sum(r['role']=='hard_negative' for r in accepted),1)
        self.assertTrue(all(r['exclusion_reason'] for r in LOCALISATION_SOURCES if r['role']=='excluded'))
        protocol=json.loads((ROOT/'configs/uk_insulator_localisation_prospective_v1.json').read_text())
        self.assertEqual(protocol['operating_scores'],[.05,.25])
        self.assertEqual(protocol['evaluation_ious'],[.3,.5])
        self.assertIn('not UK material claims',protocol['claim_boundary'])

    def test_localisation_tiling_matching_and_fusion_are_deterministic(self):
        self.assertEqual(axis_starts(640,320,.25),[0,240,320])
        epri=[{'xyxy':[0,0,10,10],'raw_score':.1,'source_model':'epri'}]
        mpid=[{'xyxy':[1,0,11,10],'raw_score':.99,'source_model':'mpid'},
              {'xyxy':[20,0,30,10],'raw_score':.2,'source_model':'mpid'}]
        fused=fixed_priority_fusion(epri,mpid,.5)
        self.assertEqual([p['source_model'] for p in fused],['epri','mpid'])
        counts=match_counts(fused,[[0,0,10,10],[20,0,30,10]],.3)
        self.assertEqual((counts['tp'],counts['fp'],counts['fn']),(2,0,0))


if __name__=='__main__':unittest.main()
