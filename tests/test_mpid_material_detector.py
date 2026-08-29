import json
import hashlib
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

from prepare_mpid_material import remap_label,split_origins
from roihu_uk_prospective_infer import gate,iou
from material_head_v2_common import decide_v2,diagnostic_counts


class MPIDMaterialDetectorTests(unittest.TestCase):
    def test_label_remap_preserves_geometry(self):
        text,count=remap_label('0 0.5 0.4 0.2 0.1\n',2)
        self.assertEqual(count,1)
        self.assertEqual(text,'2 0.5000000000 0.4000000000 0.2000000000 0.1000000000\n')
        with self.assertRaises(ValueError):remap_label('1 0.5 0.4 0.2 0.1\n',0)

    def test_origin_split_is_fixed_and_nonempty(self):
        origins={'a','b','c','d','e'}
        first=split_origins(origins,.2,29);second=split_origins(origins,.2,29)
        self.assertEqual(first,second)
        self.assertEqual(len(first),1)
        self.assertTrue(first<origins)

    def test_fixed_gate_rejects_low_score_matching_box(self):
        cfg=json.loads((ROOT/'configs/uk_prospective_porcelain_v1.json').read_text())
        prediction={'class_name':'polymer_composite','raw_detector_score':.07117267698049545,
                    'xyxy':[254.796875,248.98678588867188,284.1219177246094,382.68310546875]}
        decision=gate(prediction,[prediction],cfg)
        self.assertEqual(decision['status'],'unknown')
        self.assertIn('raw score below fixed display floor',decision['reasons'])
        self.assertGreaterEqual(iou([244,218,287,399],prediction['xyxy']),.5)

    def test_freeze_discloses_same_asset_limit(self):
        freeze=json.loads((ROOT/'data/uk_material_eval_v1/prospective_freeze_770272.json').read_text())
        self.assertFalse(freeze['model_inference_performed_on_this_image_before_freeze'])
        self.assertTrue(freeze['sibling_image_from_same_asset_group_already_inferred'])
        self.assertFalse(freeze['independent_asset_evaluation'])
        self.assertEqual(freeze['references'][0]['material'],'porcelain_ceramic')

    def test_completed_run_target_manifest_is_immutable(self):
        path=ROOT/'data/uk_material_eval_v1/mpid_run_uk_targets_v1.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'b27b9d5d43ef323b7755640499337ebe8637883d451c9b675255c93f10a1a2a8')
        self.assertEqual(len(json.loads(path.read_text())['images']),8)

    def test_report_is_english_and_has_mouse_navigation(self):
        template=(ROOT/'templates/mpid_material_report.html').read_text()
        self.assertIn('Previous image',template)
        self.assertIn('Next image',template)
        self.assertIn('Frozen two-stage crop check',template)
        self.assertIn('twoStage',template)
        self.assertIn('Four-class transfer and resolution audit',template)
        self.assertIn('resolutionOverlay',template)
        self.assertIn('Leave-one-asset-out adaptation',template)
        self.assertNotRegex(template,r'[\u4e00-\u9fff]')

    def test_two_stage_protocol_pins_complete_and_partial_crops(self):
        cfg=json.loads((ROOT/'configs/uk_two_stage_porcelain_v1.json').read_text())
        self.assertEqual(cfg['crops'][0]['xyxy'],[244,218,287,399])
        self.assertEqual(cfg['crops'][0]['expected_material'],'porcelain')
        self.assertEqual(len(cfg['crops']),3)
        self.assertEqual(cfg['classes'],['glass','porcelain','other'])
        self.assertIn('no polymer/composite class',cfg['claim_boundary'])

    def test_v2_material_gate_rejects_ood_and_accepts_supported_polymer(self):
        cfg=json.loads((ROOT/'configs/material_head_v2.json').read_text())
        thresholds={'margin':[.5]*4,'similarity':[.6]*4}
        supported=decide_v2([0,0,2,0],[0,0,1.8,0],[.1,.2,.8,.1],[.1,.2,.75,.1],[0,0,32,64],cfg,thresholds)
        self.assertEqual(supported['material'],'polymer_composite')
        ood=decide_v2([2,0,0,0],[2,0,0,0],[.3,.1,.1,.2],[.4,.1,.1,.2],[0,0,32,64],cfg,thresholds)
        self.assertEqual(ood['material'],'unknown')
        self.assertIn('outside development embedding support',ood['reasons'])

    def test_v2_uk_manifest_is_frozen_and_excluded_from_training(self):
        target=ROOT/'data/uk_material_eval_v1/material_head_v2_targets.json'
        cfg=json.loads((ROOT/'configs/material_head_v2.json').read_text())
        payload=json.loads(target.read_text())
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),cfg['uk_targets_sha256'])
        self.assertFalse(payload['training_use'])
        self.assertFalse(payload['model_v2_inference_performed_before_freeze'])
        self.assertTrue(payload['earlier_model_outputs_exist'])
        self.assertEqual(sum(len(row['boxes']) for row in payload['images']),13)
        self.assertEqual(diagnostic_counts([])['material_targets'],0)

    def test_resolution_intervention_preserves_exact_boxes_and_provenance(self):
        path=ROOT/'data/uk_material_eval_v1/material_head_v2_resolution_8090535.json'
        cfg=json.loads((ROOT/'configs/material_head_v2_resolution_8090535.json').read_text())
        payload=json.loads(path.read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),cfg['intervention_sha256'])
        self.assertFalse(payload['high_resolution_model_inference_performed_before_freeze'])
        self.assertEqual(payload['high_resolution']['width'],4*payload['low_resolution']['width'])
        self.assertEqual(payload['high_resolution']['height'],4*payload['low_resolution']['height'])
        for box in payload['boxes']:
            self.assertEqual(box['xyxy'],[4*value for value in box['xyxy_low_resolution']])
        self.assertEqual(payload['licence'],'CC BY-SA 2.0')
        self.assertIn('commons.wikimedia.org',payload['commons_file_page_url'])

    def test_loao_protocol_predeclares_every_asset_once(self):
        cfg=json.loads((ROOT/'configs/material_head_v2_loao.json').read_text())
        targets=json.loads((ROOT/'data/uk_material_eval_v1/material_head_v2_targets.json').read_text())
        expected={(row['record_id'],row['asset_group']) for row in targets['images']}
        actual={(row['test_record_id'],row['test_asset_group']) for row in cfg['folds']}
        self.assertEqual(actual,expected)
        self.assertEqual(len(cfg['folds']),3)
        self.assertEqual(cfg['adaptation']['trainable_parameters'],'final linear layer only')
        self.assertIn('never select a best fold',cfg['protocol'].lower())


if __name__=='__main__':unittest.main()
