import json
import re
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from keen_component_metrics import match_image, class_ap, summarize, validate_predictions, nms, geometric_confusion
from prepare_keen_components import polygon_box, SPLITS, digest
from build_keen_components_report import check_completed, output_counts, review_queue
from roihu_keen_components import verify_predictions
from build_keen_components_english import build as build_english, link_existing, english_results
from audit_keen_development import miss_reason, review_needs, trace_publisher_labels
from paper_material_demo import material_decision, extent, suppress, runtime as model_runtime
from prepare_paper_selection import selection_rows


def prediction(c=0, score=.9, box=None):
    return dict(class_id=c, score=score, box=box or [0, 0, 10, 10])


class ComponentMetricTests(unittest.TestCase):
    def test_material_inference_rejects_local_devices_before_loading_torch(self):
        with patch.dict('os.environ', {}, clear=True):
            for device in ['cpu', 'mps', 'cuda']:
                with self.assertRaisesRegex(RuntimeError, 'gputest'):
                    model_runtime(device)
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'example', 'SLURM_JOB_PARTITION': 'other'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'gputest'):
                model_runtime('cuda')

    def test_public_demo_selection_preserves_split_and_original_annotation_units(self):
        sources = {s: {'images': [{'id': 1, 'file_name': 'one.jpg'}],
                       'annotations': [{'image_id': 1, 'category_id': c, 'id': n} for n, c in enumerate(classes)]}
                   for s, classes in [('train', [2, 2]), ('test', [1])]}
        before = deepcopy(sources)
        rows = selection_rows(sources, {'train': ['one.jpg'], 'test': ['one.jpg']})
        self.assertEqual([len(r['annotations']) for r in rows], [2, 1])
        self.assertEqual([r['split'] for r in rows], ['train', 'test'])
        self.assertEqual(sources, before)
        sources['train']['images'].append(sources['train']['images'][0])
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            selection_rows(sources, {'train': ['one.jpg']})

    def test_material_demo_rejects_tiny_disagreeing_and_ambiguous_crops(self):
        large=[0,0,100,100]
        self.assertEqual(material_decision([.8,.3],[.7,.4],large)[0],'glass')
        self.assertEqual(material_decision([.3,.8],[.4,.7],large)[0],'porcelain')
        self.assertEqual(material_decision([.8,.3],[.7,.4],[0,0,5,100])[0],'unknown')
        self.assertEqual(material_decision([.8,.3],[.4,.7],large)[0],'unknown')
        self.assertEqual(material_decision([.8,.79],[.8,.79],large)[0],'unknown')

    def test_material_demo_crop_clips_without_upscaling(self):
        self.assertEqual(extent([1.5,2.5,20.1,30.9],25,32,0),[1,2,21,31])
        self.assertEqual(extent([1.5,2.5,20.1,30.9],25,32,.2),[0,0,24,32])

    def test_material_demo_nms_preserves_raw_candidates_and_higher_score(self):
        low=prediction(score=.2); high=prediction(score=.8); separate=prediction(score=.4,box=[20,20,30,30])
        candidates=[low,high,separate];before=deepcopy(candidates)
        self.assertEqual(suppress(candidates,.5),[high,separate]);self.assertEqual(candidates,before)

    def test_material_demo_viewer_distinguishes_source_annotations_from_predictions(self):
        template=(ROOT/'templates/paper_material_demo.html').read_text()
        self.assertNotRegex(template,r'[\u3400-\u9fff]')
        self.assertIn('lang="en-GB"',template)
        self.assertIn('Publisher polygons',template)
        self.assertIn('This is not a verified negative image.',template)
        self.assertIn('Material similarities are uncalibrated.',template)
        self.assertIn('material_verified', (ROOT/'scripts/paper_material_demo.py').read_text())

    def test_source_label_trace_reproduces_geometry_without_relabelling(self):
        polygon=[{'x':0,'y':0},{'x':10,'y':0},{'x':10,'y':10}]
        label={'objects':[{'value':'crossarm','polygon':polygon}]}
        row={'width':100,'height':100,'references':[{'annotation_id':'sample_0','class_name':'crossarm',
             'polygon':polygon,'box':[0,0,10,10]}]}
        before=deepcopy((label,row))
        traced=trace_publisher_labels(label,row,[0])
        self.assertEqual(traced[0]['publisher_value'],'crossarm')
        self.assertEqual(traced[0]['expert_review_status'],'PENDING')
        self.assertFalse(traced[0]['original_label_changed'])
        self.assertEqual((label,row),before)
        row['references'][0]['class_name']='insulator'
        with self.assertRaisesRegex(ValueError,'publisher object'):
            trace_publisher_labels(label,row,[0])

    def test_development_audit_distinguishes_score_assignment_and_suppression(self):
        ref = prediction(c=1)
        low = prediction(c=1, score=.1)
        high = prediction(c=1, score=.8)
        self.assertEqual(miss_reason(ref, [low], [low])['reason'], 'low_score_overlap')
        self.assertFalse(miss_reason(ref, [low], [low])['recoverable_true_positive_claim'])
        self.assertEqual(miss_reason(ref, [high], [high])['reason'], 'assignment_conflict')
        self.assertEqual(miss_reason(ref, [], [high])['reason'], 'suppressed_overlap')

    def test_development_audit_distinguishes_wrong_class_partial_and_missing_support(self):
        ref = prediction(c=1)
        wrong = prediction(c=2)
        partial = prediction(c=1, box=[5,0,15,10])
        self.assertEqual(miss_reason(ref, [wrong], [wrong])['reason'], 'wrong_class_overlap')
        self.assertEqual(miss_reason(ref, [partial], [partial])['reason'], 'partial_overlap')
        self.assertEqual(miss_reason(ref, [], [])['reason'], 'no_overlap')

    def test_uk_review_priority_never_creates_labels_or_drops_empty_images(self):
        row = dict(image_id='empty', sha256='pixels', width=100, height=100, title='Source',
                   credit='Author', source_page='source', license='CC', license_url='licence',
                   material_diagnostics=[], predictions={'dino_hardware':[]})
        task = review_needs(row)
        self.assertEqual(task['image_id'], 'empty')
        self.assertEqual(task['review_status'], 'UNREVIEWED')
        self.assertEqual(task['native_gate_pass'], 0)
        self.assertEqual(task['material_labels_verified'], 0)
        self.assertIsNone(task['asset_id'])
        self.assertFalse(task['training_approved'])
        self.assertFalse(task['priority_is_accuracy_or_suitability'])

    def test_english_template_preserves_existing_controls_and_translates_dynamic_text(self):
        original = (ROOT / 'templates/keen_components_report.html').read_text()
        english = (ROOT / 'templates/keen_components_report_en.html').read_text()
        self.assertNotRegex(english, r'[\u3400-\u9fff]')
        self.assertIn('lang="en-GB"', english)
        original_ids = set(re.findall(r'\bid="([^"]+)"', original))
        english_ids = set(re.findall(r'\bid="([^"]+)"', english))
        self.assertTrue(original_ids <= english_ids)
        for body in (original, english):
            toggles = re.findall(r'class="classToggle" value="(\d)" checked', body)
            self.assertEqual(toggles, ['0', '1', '2'])
        self.assertIn('const m=uk?null:match(ps,r.references,s.threshold)', english)
        self.assertIn('Display filters do not change scoring.', english)
        self.assertIn('not a calibrated probability', english)

    def test_english_presentation_cannot_overwrite_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'frozen'
            root.mkdir()
            for target in (root, root / 'report_en', root.parent):
                with self.assertRaisesRegex(ValueError, 'separate directory'):
                    build_english(root, target)

    def test_presentation_links_are_idempotent_but_never_replace_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw.json'
            source.write_text('original')
            target = root / 'linked.json'
            link_existing(target, source)
            link_existing(target, source)
            self.assertEqual(target.read_text(), 'original')
            with self.assertRaisesRegex(ValueError, 'retarget'):
                link_existing(target, root / 'different.json')
            with self.assertRaisesRegex(ValueError, 'replace existing'):
                link_existing(source, target)

    def test_english_summary_uses_saved_metrics_without_relabelling_uk_counts(self):
        source = ROOT / 'runs/keen_components/epri_components_v1_20260827/report/data.json'
        if not source.exists():
            self.skipTest('Completed experiment is not installed')
        report = english_results(json.loads(source.read_text()))
        self.assertNotRegex(report, r'[\u3400-\u9fff]')
        self.assertIn('11 TP, 9 FP, 89 FN; recall 11.0%', report)
        self.assertIn('20/27 UK images have no supervised output', report)
        self.assertIn('All 216 material proposals remain unknown', report)

    def test_wrong_class_is_fp_and_fn_even_when_geometry_is_perfect(self):
        m = match_image([prediction(1)], [prediction(0)])
        self.assertEqual((m["tp"],m["fp"],m["fn"]), (0,1,1))

    def test_nested_component_classes_survive_nms_and_score_independently(self):
        ps = [prediction(0), prediction(1), prediction(2)]
        self.assertEqual(len(nms(ps)),3)
        self.assertEqual(match_image(nms(ps), ps)["tp"],3)

    def test_duplicate_box_cannot_match_twice(self):
        m = match_image([prediction(), prediction(score=.8)], [prediction()])
        self.assertEqual((m["tp"],m["fp"],m["fn"]),(1,1,0))

    def test_ap_perfect_missed_and_absent_classes(self):
        rows = [dict(image_id="a", references=[prediction()], predictions=[prediction()])]
        self.assertAlmostEqual(class_ap(rows,0),1.)
        self.assertIsNone(class_ap(rows,1))
        rows[0]["predictions"] = []
        self.assertEqual(class_ap(rows,0),0.)

    def test_earlier_false_positive_reduces_ap(self):
        ps = [prediction(score=.95,box=[20,20,30,30]),prediction(score=.8)]
        rows = [dict(image_id="a", references=[prediction()], predictions=ps)]
        self.assertAlmostEqual(class_ap(rows,0),.5)

    def test_no_matches_across_images(self):
        rows = [dict(image_id="a", references=[prediction()], predictions=[]),
                dict(image_id="b", references=[], predictions=[prediction()])]
        self.assertEqual(class_ap(rows,0),0.)
        self.assertEqual(summarize(rows,["pole"])["operating_points"]["0.25"]["micro"]["tp"],0)

    def test_duplicate_image_records_rejected(self):
        r = dict(image_id="a",references=[],predictions=[])
        with self.assertRaises(ValueError): summarize([r,r],["pole"])

    def test_invalid_boxes_scores_classes_rejected(self):
        for p in [prediction(3),prediction(score=float('nan')),prediction(box=[0,0,101,10]),prediction(box=[1,1,1,3])]:
            with self.assertRaises(ValueError): validate_predictions([p],100,100,3)

    def test_geometry_confusion_does_not_hide_class_error(self):
        rows = [dict(image_id='a',references=[prediction(0)],predictions=[prediction(1)])]
        self.assertEqual(geometric_confusion(rows,['pole','crossarm'])['matrix'][0][1],1)
        self.assertEqual(match_image(rows[0]['predictions'],rows[0]['references'])['tp'],0)

    def test_polygon_bounds_clip_without_mutating_source(self):
        ps = [dict(x=-1,y=2),dict(x=11,y=2),dict(x=5,y=12)]
        box,clipped=polygon_box(ps,10,10)
        self.assertEqual(box,[0,2,10,10]); self.assertTrue(clipped)
        self.assertEqual(ps[0]['x'],-1)
        with self.assertRaises(ValueError): polygon_box(ps[:2],10,10)

    def test_protocol_pins_selection_and_preserves_circuit_boundary(self):
        p=json.loads((ROOT/'configs/keen_components_v1.json').read_text())
        self.assertEqual(p['classes'],['pole','crossarm','insulator'])
        self.assertFalse(p['baseline_prompt_search'])
        self.assertFalse(p['frozen_uk_holdouts_used'])
        groups=[c for rows in SPLITS.values() for c in rows]
        self.assertEqual(len(groups),len(set(groups)))
        self.assertRegex(p['selection_plan_sha256'],r'^[0-9a-f]{64}$')
        local_plan=ROOT/'data/external/epri_components_v1/selection_plan.json'
        if local_plan.exists():
            self.assertEqual(p['selection_plan_sha256'],digest(local_plan))

    def test_qualitative_counts_do_not_claim_accuracy(self):
        rows=[{"predictions":{"supervised":[prediction(1,score=.1)]}},
              {"predictions":{"supervised":[]}}]
        counts=output_counts(rows,"supervised",["pole","crossarm","insulator"])
        self.assertEqual(counts["0.25"]["images_without_output"],2)
        self.assertEqual(counts["0.05"]["boxes_by_class"]["crossarm"],1)
        self.assertTrue(counts["0.05"]["not_accuracy"])

    def test_review_queue_retains_empty_images_and_unknown_material(self):
        def row(key,ps):
            return {"image_id":key,"image_file":key+".jpg","sha256":"pixels","source_page":"source",
                    "credit":"author","license":"CC BY-SA","license_url":"license",
                    "predictions":{"supervised":ps}}
        queue=review_queue([row("empty",[]),row("candidate",[prediction(2)])],"checkpoint")
        self.assertEqual([x["image_id"] for x in queue["image_tasks"]],["empty","candidate"])
        self.assertEqual(queue["image_tasks"][0]["machine_proposals"],[])
        self.assertEqual(queue["image_tasks"][1]["reviewed_objects"],[])
        self.assertIsNone(queue["proposals"][0]["material_label"])

    def test_report_rejects_incomplete_budget_or_evaluation_selection(self):
        report={"status":"COMPLETED_MULTICOMPONENT_TRAINING_AND_FROZEN_EVALUATION",
                "training_progress":{"completed_epochs":40},"dataset_manifest_sha256":"data",
                "uk_manifest_sha256":"uk","selected_checkpoint_sha256":"trained",
                "config":{"training":{"epochs":40},"checkpoint_sha256":"base",
                          "baseline_prompts":["pole"],"evaluation":{"score_thresholds":[.25]}}}
        choices={"protocol_sha256":"protocol","dataset_manifest_sha256":"data","uk_manifest_sha256":"uk",
                 "checkpoint_sha256":"trained","baseline_checkpoint_sha256":"base","prompts":["pole"],
                 "evaluation":{"score_thresholds":[.25]},"eval_used_for_selection":False,"uk_used_for_selection":False}
        check_completed(report,choices,"protocol")
        incomplete=deepcopy(report);incomplete["training_progress"]["completed_epochs"]=39
        with self.assertRaises(ValueError):check_completed(incomplete,choices,"protocol")
        selected=deepcopy(choices);selected["eval_used_for_selection"]=True
        with self.assertRaises(ValueError):check_completed(report,selected,"protocol")
        failed=deepcopy(report);failed["status"]="FAILED"
        with self.assertRaises(ValueError):check_completed(failed,choices,"protocol")

    def test_raw_verifier_detects_tampering_and_incorrect_postprocessing(self):
        row={"image_id":"a","sha256":"pixels","width":100,"height":100,"references":[]}
        payload={"image_id":"a","image_sha256":"pixels","arm":"supervised",
                 "raw_predictions":[prediction()],"predictions":[prediction()]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"a.json";path.write_text(json.dumps(payload))
            record={"image_id":"a","arm":"supervised","prediction_file":"a.json",
                    "prediction_sha256":digest(path),"metrics_025":None}
            self.assertEqual(verify_predictions(root,[row],[record],["pole"],.001),1)
            payload["predictions"]=[];path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"changed"):
                verify_predictions(root,[row],[record],["pole"],.001)
            record["prediction_sha256"]=digest(path)
            with self.assertRaisesRegex(ValueError,"NMS"):
                verify_predictions(root,[row],[record],["pole"],.001)


if __name__ == '__main__': unittest.main()
