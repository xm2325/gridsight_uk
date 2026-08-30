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
from acquire_uk_insulator_localisation_v2 import SOURCES as LOCALISATION_V2_SOURCES
from acquire_uk_insulator_localisation_v3 import SOURCES as LOCALISATION_V3_SOURCES
from prepare_uk_insulator_adaptation_v1 import POSITIVES as ADAPTATION_POSITIVES
from prepare_uk_insulator_adaptation_v1 import NEGATIVES as ADAPTATION_NEGATIVES
from prepare_uk_insulator_adaptation_v1 import square_crop
from prepare_uk_insulator_adaptation_v2 import crop_labels as crop_labels_v2
from prepare_uk_insulator_adaptation_v2 import square_crop as square_crop_v2
from prepare_uk_insulator_adaptation_v2 import verify_definitions as verify_adaptation_v2
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

    def test_second_uk_acceptance_is_frozen_before_adaptation(self):
        accepted=[r for r in LOCALISATION_V2_SOURCES if r['role'] in {'prospective_test','hard_negative'}]
        self.assertEqual(len(accepted),5)
        self.assertEqual(sum(len(r['boxes']) for r in accepted),14)
        self.assertEqual(len({r['asset_group'] for r in accepted}),5)
        self.assertEqual(sum(r['role']=='hard_negative' for r in accepted),1)
        self.assertTrue(all(r['exclusion_reason'] for r in LOCALISATION_V2_SOURCES if r['role']=='excluded'))
        manifest=json.loads((ROOT/'data/external/uk_insulator_localisation_v2/manifest.json').read_text())
        self.assertTrue(manifest['selection_frozen_before_adapted_model_inference'])
        self.assertFalse(manifest['model_inference_performed_before_freeze'])
        self.assertFalse(manifest['prior_manifest_image_hash_overlap'])

    def test_uk_adaptation_definitions_are_group_disjoint_from_acceptance(self):
        self.assertEqual(len(ADAPTATION_POSITIVES),7)
        self.assertEqual(sum(len(v['boxes']) for v in ADAPTATION_POSITIVES.values()),21)
        self.assertEqual(len(ADAPTATION_NEGATIVES),3)
        train={k for k,v in ADAPTATION_POSITIVES.items() if v['split']=='train'} | {
            k for k,v in ADAPTATION_NEGATIVES.items() if v=='train'}
        dev={k for k,v in ADAPTATION_POSITIVES.items() if v['split']=='dev'} | {
            k for k,v in ADAPTATION_NEGATIVES.items() if v=='dev'}
        self.assertFalse(train & dev)
        pilot=json.loads((ROOT/'data/external/uk_distribution_pilot_v1/manifest.json').read_text())
        by_id={r['geograph_id']:r for r in pilot['images']}
        adaptation_hashes={by_id[key]['sha256'] for key in train|dev}
        acceptance=json.loads((ROOT/'data/external/uk_insulator_localisation_v2/manifest.json').read_text())
        acceptance_hashes={r['image_sha256'] for r in acceptance['records'] if r['role']!='excluded'}
        self.assertFalse(adaptation_hashes & acceptance_hashes)
        self.assertEqual(square_crop([[90,90,100,100]],100,100,1.5),[0,0,100,100])
        self.assertEqual(square_crop([[10,20,20,30]],200,100,1.5),[0,0,200,100])
        protocol=json.loads((ROOT/'configs/uk_insulator_adaptation_v1.json').read_text())
        self.assertEqual(protocol['training']['epochs'],10)
        self.assertEqual(protocol['evaluation']['operating_scores'],[.05,.25])
        self.assertEqual(protocol['evaluation']['evaluation_ious'],[.3,.5])
        self.assertIn('does not replace',protocol['model_role'])
        self.assertIn('No acceptance-selected retry',protocol['budget'])

    def test_third_uk_acceptance_is_frozen_source_preserved_and_unconsumed(self):
        accepted=[r for r in LOCALISATION_V3_SOURCES if r['role'] in {'prospective_test','hard_negative'}]
        excluded=[r for r in LOCALISATION_V3_SOURCES if r['role']=='excluded']
        self.assertEqual(len(accepted),9)
        self.assertEqual(sum(len(r['boxes']) for r in accepted),30)
        self.assertEqual(len({r['asset_group'] for r in accepted}),9)
        self.assertEqual(sum(r['role']=='hard_negative' for r in accepted),2)
        self.assertTrue(all(r['exclusion_reason'] for r in excluded))
        manifest=json.loads((ROOT/'data/external/uk_insulator_localisation_v3/manifest.json').read_text())
        manifest_accepted=[r for r in manifest['records'] if r['role']!='excluded']
        self.assertTrue(all(r['author'] and r['photo_page_url'] and r['licence_url'] for r in manifest_accepted))
        self.assertTrue(all(r['negative_evidence'] for r in manifest_accepted if r['role']=='hard_negative'))
        self.assertTrue(manifest['selection_frozen_before_v2_adapted_model_inference'])
        self.assertFalse(manifest['model_inference_performed_before_freeze'])
        self.assertEqual(manifest['prior_overlap'],{'image_hashes':{},'photo_ids':{},'asset_groups':{}})

    def test_adaptation_v2_definitions_keep_v3_out_and_crops_label_safe(self):
        sources,groups=verify_adaptation_v2()
        self.assertEqual((sum(r['split']=='train' for r in sources),len(groups['train'])),(15,13))
        self.assertEqual((sum(r['split']=='dev' for r in sources),len(groups['dev'])),(9,9))
        self.assertEqual(sum(len(r['boxes']) for r in sources if r['split']=='train'),60)
        self.assertEqual(sum(len(r['boxes']) for r in sources if r['split']=='dev'),18)
        self.assertFalse(groups['train'] & groups['dev'])
        crop=square_crop_v2([90,40,100,50],100,300,8)
        self.assertEqual(crop,[0,0,100,100])
        self.assertTrue(0 <= crop[0] < crop[2] <= 100)
        self.assertTrue(0 <= crop[1] < crop[3] <= 300)
        self.assertIsNone(crop_labels_v2([[10,10,30,30]], [20,0,40,40]))
        self.assertEqual(crop_labels_v2([[10,10,30,30]], [0,0,40,40]),[[10,10,30,30]])
        development=json.loads((ROOT/'data/external/uk_insulator_development_v2/manifest.json').read_text())
        acceptance=json.loads((ROOT/'data/external/uk_insulator_localisation_v3/manifest.json').read_text())
        self.assertEqual(development['v3_acceptance_manifest_sha256'],
                         'd74f206e506c9c61303cdf20c092c44c107332cc3931ccf0f6a8079e68ac50ac')
        self.assertFalse(development['v3_acceptance_read_for_training'])
        self.assertFalse(development['records'][0]['asset_group'] in
                         {r['asset_group'] for r in acceptance['records'] if r['role']!='excluded'})
        protocol=json.loads((ROOT/'configs/uk_insulator_adaptation_v2.json').read_text())
        self.assertEqual(protocol['dataset_manifest_sha256'],
                         'a61545de5e83f574f1faa459e2c1976c091ab40f7569185a1ad9b745bbdae6ed')
        self.assertEqual(protocol['acceptance_manifest_sha256'],
                         'd74f206e506c9c61303cdf20c092c44c107332cc3931ccf0f6a8079e68ac50ac')
        self.assertEqual(protocol['baseline_checkpoint_sha256'],
                         '30ecb12f6fa7736af950075a12e5a48cebec25223a0fe71f0a7776a92535ddc7')
        self.assertEqual(protocol['training']['epochs'],8)
        self.assertEqual(protocol['evaluation']['operating_scores'],[.05,.25])
        self.assertEqual(protocol['evaluation']['evaluation_ious'],[.3,.5])
        self.assertIn('No acceptance-selected retry',protocol['budget'])

    def test_uk_localisation_result_preserves_failures_and_claim_boundary(self):
        result_path=ROOT/'runs/uk_insulator_localisation/v1_20260830/results.json'
        report_path=ROOT/'runs/uk_capabilities/v3_20260827/report/localisation_prospective/data.json'
        if not result_path.exists() or not report_path.exists():
            self.skipTest('Optional prospective localisation output is not in the source release')
        result=json.loads(result_path.read_text());report=json.loads(report_path.read_text())
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['integrity']['training_or_parameter_updates'],0)
        self.assertFalse(result['integrity']['thresholds_selected_from_acceptance_results'])
        self.assertFalse(result['integrity']['mpid_material_classes_scored_as_uk_material_truth'])
        primary={name:result['metrics'][name]['0.05']['0.3'] for name in result['protocol']['arms']}
        self.assertEqual((primary['epri_full']['tp'],primary['epri_full']['fp']),(1,0))
        self.assertEqual((primary['epri_full_plus_tiles']['tp'],primary['epri_full_plus_tiles']['fp']),(2,0))
        self.assertEqual((primary['proposal_fusion']['tp'],primary['proposal_fusion']['fp']),(4,22))
        self.assertEqual(len(report['gallery']),8)
        self.assertIn('Target-domain supervised localisation is required',report['conclusion'])

    def test_uk_adaptation_result_and_report_preserve_prospective_boundary(self):
        result_path=ROOT/'runs/uk_insulator_adaptation/v1_20260830/results.json'
        report_path=ROOT/'runs/uk_capabilities/v3_20260827/report/localisation_adaptation/data.json'
        if not result_path.exists() or not report_path.exists():
            self.skipTest('Optional UK adaptation output is not in the source release')
        result=json.loads(result_path.read_text());report=json.loads(report_path.read_text())
        self.assertEqual(result['status'],'COMPLETE')
        self.assertFalse(result['integrity']['acceptance_used_for_training_or_checkpoint_selection'])
        self.assertFalse(result['integrity']['thresholds_selected_from_acceptance_results'])
        self.assertEqual(result['integrity']['acceptance_inference_passes_per_checkpoint'],1)
        self.assertFalse(result['integrity']['outputs_are_calibrated_probabilities'])
        baseline=result['metrics']['baseline_epri_full_plus_tiles']['0.05']['0.3']
        adapted=result['metrics']['adapted_specialist_full_plus_tiles']['0.05']['0.3']
        self.assertEqual((baseline['tp'],baseline['fp'],baseline['fn']),(0,1,14))
        self.assertEqual((adapted['tp'],adapted['fp'],adapted['fn']),(5,3,9))
        self.assertEqual(len(report['gallery']),5)
        self.assertIn('not yet Keen-style',report['conclusion'])


if __name__=='__main__':unittest.main()
