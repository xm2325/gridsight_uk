import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from uk_review_common import pole_regions,map_crop_predictions,validate_review_image,validate_review_v3
from serve_uk_review import DraftStore,RevisionConflict,ReviewHandler
from uk_capability_common import crop_extent,material_quality,diagnostic_decision


class UKReviewTests(unittest.TestCase):
    def test_v3_inspection_region_has_no_detector_confidence(self):
        o={'id':'r','class_id':4,'box':[1,2,30,40],'origin':'derived_geometry','entity_kind':'inspection_region','material':None}
        self.draft['objects']=[o];validate_review_v3(self.draft,self.source)
        o['score']=.9
        with self.assertRaisesRegex(ValueError,'confidence'):validate_review_v3(self.draft,self.source)
        o.pop('score');o['entity_kind']='component'
        with self.assertRaisesRegex(ValueError,'semantics'):validate_review_v3(self.draft,self.source)

    def test_v3_steelwork_evidence_is_required_before_second_review(self):
        o={'id':'s','class_id':3,'box':[1,2,30,40],'origin':'machine_proposal','entity_kind':'component','material':None}
        self.draft.update(objects=[o],reviewer='QA',status='ready_for_second_review')
        with self.assertRaisesRegex(ValueError,'Steelwork requires'):validate_review_v3(self.draft,self.source)
        o['steelwork_evidence']='Test evidence only';validate_review_v3(self.draft,self.source)

    def test_v3_parent_must_be_a_pole_in_same_image(self):
        o={'id':'s','class_id':3,'box':[1,2,30,40],'origin':'manual_draft','entity_kind':'component','material':None,'parent_pole_id':'missing'}
        self.draft['objects']=[o]
        with self.assertRaisesRegex(ValueError,'Parent'):validate_review_v3(self.draft,self.source)
        self.draft['objects'].append({'id':'missing','class_id':0,'box':[1,2,50,100],'origin':'manual_draft','entity_kind':'component','material':None})
        validate_review_v3(self.draft,self.source)

    def test_v3_export_preserves_region_semantics_and_does_not_migrate_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=DraftStore(tmp,[self.source],validate_review_v3,'gridsight-uk-review-v3')
            store.save_image('a',self.draft,0)
            exported=store.export();self.assertEqual(exported['region_class_ids'],[4]);self.assertFalse(exported['training_approved'])
            with self.assertRaisesRegex(ValueError,'schema'):DraftStore(tmp,[self.source]).load()

    def test_v3_interface_and_guides_are_english(self):
        for file in ['templates/uk_component_review_v3.html','UK_COMPONENT_ANNOTATION_GUIDE_EN.md','KEEN_CAPABILITY_DESIGN_EN.md']:
            self.assertNotRegex((ROOT/file).read_text(),r'[\u3400-\u9fff]')

    def test_material_context_crop_preserves_native_bounds(self):
        self.assertEqual(crop_extent([10,20,30,40],100,80,.25),[5,15,35,45])
        self.assertEqual(crop_extent([0,0,30,40],32,42,.25),[0,0,32,42])

    def test_material_pixel_gate_uses_object_not_padded_crop(self):
        cfg=json.loads((ROOT/'configs/uk_capabilities_v3.json').read_text())
        self.assertFalse(material_quality([0,0,8,100],cfg))
        self.assertTrue(material_quality([0,0,32,32],cfg))

    def test_material_rankings_always_abstain_without_target_validation(self):
        labels=['glass','porcelain','polymer','metal_fitting']
        for a,b,reason in [([.8,.1,.1,.1],[.7,.1,.1,.1],'uncalibrated_no_target_validation'),
            ([.8,.1,.1,.1],[.1,.8,.1,.1],'crop_context_disagreement'),
            ([.1,.1,.1,.8],[.1,.1,.1,.8],'non_insulator_hypothesis')]:
            d=diagnostic_decision(a,b,labels)
            self.assertEqual(d['material'],'unknown');self.assertFalse(d['accepted'])
            self.assertEqual(d['reason'],reason);self.assertFalse(d['scores_are_probabilities'])

    def setUp(self):
        self.cfg=json.loads((ROOT/"configs/uk_component_review_v2.json").read_text())["roi"]
        self.source={"image_id":"a","sha256":"source","width":800,"height":600,
                     "source_page":"source","credit":"author","license":"CC BY-SA","license_url":"license"}
        self.draft={"image_sha256":"source","status":"draft","reviewer":"","notes":"","objects":[]}

    def test_rois_are_bounded_automatic_and_not_pole_top_detections(self):
        ps=[{"class_id":0,"score":.8,"box":[10,10,30,500]}]
        regions=pole_regions(ps,800,600,self.cfg)
        self.assertEqual(len(regions),1)
        self.assertFalse(regions[0]["manual"])
        self.assertEqual(regions[0]["box"][:2],[0,0])
        self.assertLessEqual(regions[0]["box"][2],800)

    def test_low_confidence_and_small_poles_do_not_drive_rois(self):
        ps=[{"class_id":0,"score":.1,"box":[10,10,30,500]},
            {"class_id":0,"score":.9,"box":[10,10,20,20]}]
        self.assertEqual(pole_regions(ps,800,600,self.cfg),[])

    def test_crop_coordinates_map_back_without_mutation(self):
        p={"class_id":2,"score":.5,"box":[5,6,20,22]}
        result=map_crop_predictions([p],[100,200,300,400],800,600)
        self.assertEqual(result[0]["box"],[105,206,120,222])
        self.assertEqual(p["box"],[5,6,20,22])

    def test_material_claim_requires_evidence_and_correct_object_type(self):
        obj={"id":"x","class_id":2,"box":[1,2,30,40],"origin":"machine_proposal","material":"porcelain"}
        self.draft["objects"]=[obj]
        with self.assertRaisesRegex(ValueError,"evidence"):validate_review_image(self.draft,self.source)
        obj["material"]="unknown";validate_review_image(self.draft,self.source)
        obj["class_id"]=0
        with self.assertRaisesRegex(ValueError,"insulators"):validate_review_image(self.draft,self.source)

    def test_invalid_geometry_and_training_approval_rejected(self):
        obj={"id":"x","class_id":0,"box":[1,2,30,40],"origin":"manual_draft","material":None}
        self.draft["objects"]=[obj]
        for bad in ([1,2,900,40],[1,float("nan"),20,40],[1,2,1,4]):
            obj["box"]=bad
            with self.assertRaises(ValueError):validate_review_image(self.draft,self.source)
        obj["box"]=[1,2,30,40];obj["training_approved"]=True
        with self.assertRaisesRegex(ValueError,"approve"):validate_review_image(self.draft,self.source)
        obj["training_approved"]=False;obj["expert_validated"]=True
        with self.assertRaisesRegex(ValueError,"expert"):validate_review_image(self.draft,self.source)

    def test_review_readiness_requires_reviewer(self):
        self.draft["status"]="ready_for_second_review"
        with self.assertRaisesRegex(ValueError,"Reviewer"):validate_review_image(self.draft,self.source)

    def test_draft_store_is_durable_versioned_and_never_training_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=DraftStore(tmp,[self.source]);saved=store.save_image("a",self.draft,0)
            self.assertEqual(saved["revision"],1)
            with self.assertRaises(RevisionConflict):store.save_image("a",self.draft,0)
            restored=DraftStore(tmp,[self.source]).export()
            self.assertFalse(restored["training_approved"])
            self.assertEqual(restored["images"]["a"]["revision"],1)
            self.assertEqual(len((Path(tmp)/"events.jsonl").read_text().splitlines()),1)

    def test_changed_image_fingerprint_cannot_be_imported(self):
        self.draft["image_sha256"]="other"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"fingerprint"):DraftStore(tmp,[self.source]).save_image("a",self.draft,0)

    def handler(self,stores,origin="http://127.0.0.1:8772",path="/api/save",revision=0):
        payload=json.dumps({"image_id":"a","draft":self.draft,"revision":revision}).encode()
        h=ReviewHandler.__new__(ReviewHandler);h.path=path;h.stores=stores;h.token="test-token"
        h.server=SimpleNamespace(server_port=8772);h.rfile=io.BytesIO(payload)
        h.headers={"Origin":origin,"X-Review-Token":"test-token","Content-Type":"application/json","Content-Length":str(len(payload))}
        result=[];h.api_json=lambda value,status=200,**kw:result.append((status,value))
        return h,result

    def test_cross_origin_write_is_rejected_even_with_token(self):
        h,res=self.handler({},origin="https://untrusted.example")
        h.do_POST();self.assertEqual(res[0][0],403)

    def test_missing_csrf_token_is_rejected(self):
        h,res=self.handler({});h.headers["X-Review-Token"]=""
        h.do_POST();self.assertEqual(res[0][0],403)

    def test_qa_edits_are_isolated_and_stale_revision_returns_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            stores={key:DraftStore(Path(tmp)/key,[self.source]) for key in ["qa","main"]}
            h,res=self.handler(stores,path="/api/save?qa=1");h.do_POST()
            self.assertEqual(res[0][0],200)
            self.assertEqual(stores["main"].load()["images"]["a"]["revision"],0)
            self.assertEqual(stores["qa"].load()["images"]["a"]["revision"],1)
            h,res=self.handler(stores,path="/api/save?qa=1");h.do_POST()
            self.assertEqual(res[0][0],409)

    def test_request_cannot_insert_training_approval_into_image(self):
        self.draft["training_approved"]=True
        with tempfile.TemporaryDirectory() as tmp:
            store=DraftStore(tmp,[self.source])
            h,res=self.handler({"main":store,"qa":store});h.do_POST()
            self.assertEqual(res[0][0],400)


if __name__=="__main__":unittest.main()
