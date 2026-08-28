"""Data isolation, exact targets and training-sanity gates without CUDA."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_insplad_adaptation import family, ordered_images, split_families, yolo_labels, choose_overfit
from insplad_adapt_common import collapse_targets, overfit_gate, verify_records
from roihu_benchmark100 import operating_metrics


class AdaptationDataTests(unittest.TestCase):
    def test_family_merges_related_capture_prefixes(self):
        self.assertEqual(family("282-1_DJI_0010.jpg"), family("282-2_DJI_0030.jpg"))
        with self.assertRaises(ValueError):
            family("../282-1_DJI_0010.jpg")

    def test_split_excludes_all_validation_families(self):
        rows = [dict(file_name=f"{i}-{view}_DJI_0001.jpg") for i in range(10) for view in (1, 2)]
        val = [dict(file_name="9-3_DJI_0020.jpg")]
        protocol = dict(seed="fixed", holdout_families=2, development_families=2)
        groups, excluded = split_families(rows, val, protocol)
        self.assertEqual(excluded, ["9"])
        self.assertNotIn("9", sum(groups.values(), []))
        self.assertEqual(len(sum(groups.values(), [])), len(set(sum(groups.values(), []))))
        self.assertEqual((groups, excluded), split_families(list(reversed(rows)), val, protocol))

    def test_sampling_balances_coarse_families_deterministically(self):
        rows = [dict(file_name=f"{i}-{v}_DJI_{j:04d}.jpg") for i in range(3) for v in (1, 2) for j in range(4)]
        selected = list(ordered_images(rows, ["0", "1", "2"], "fixed"))
        self.assertEqual(selected, list(ordered_images(list(reversed(rows)), ["0", "1", "2"], "fixed")))
        self.assertEqual(len({family(r["file_name"]) for r in selected[:3]}), 3)

    def test_yolo_conversion_preserves_original_boxes_and_empty_negatives(self):
        self.assertEqual(yolo_labels([], 200, 100), "")
        values = yolo_labels([dict(box=[20, 10, 180, 90])], 200, 100).split()
        self.assertEqual(values[0], "0")
        self.assertEqual(list(map(float, values[1:])), [0.5, 0.5, 0.8, 0.8])
        for box in ([0, 0, 300, 10], [5, 5, 2, 9], [0, 0, float("nan"), 5]):
            with self.assertRaises(ValueError):
                yolo_labels([dict(box=box)], 200, 100)

    def test_overfit_uses_training_only_and_one_of_each_material(self):
        rows = [dict(image_id=i, split=role, references=[dict(category_id=cat)])
                for i, role, cat in ((0, "holdout", 8), (1, "dev", 7), (2, "train", 8), (3, "train", 7))]
        self.assertEqual(choose_overfit(rows), [2, 3])
        with self.assertRaises(ValueError):
            choose_overfit(rows[:2])

    def test_fixed_protocol_does_not_select_on_holdout(self):
        protocol = json.loads((ROOT / "configs/insplad_adapt_protocol.json").read_text())
        self.assertTrue(protocol["exclude_all_official_validation_families"])
        self.assertEqual(protocol["split_sizes"], dict(train=320, dev=80, holdout=100))
        self.assertIn("fresh original", protocol["adaptation"]["initialisation"])
        self.assertEqual(protocol["overfit"]["gate"]["required_recall"], 1.0)
        self.assertEqual(protocol["prompt_arms"]["material_names"]["target_ids"], [0, 1])

    def test_material_aliases_collapse_before_nms(self):
        predictions = [dict(class_id=c, box=[0, 0, 10, 10], score=s)
                       for c, s in ((0, 0.9), (1, 0.8), (2, 0.95))]
        targets = collapse_targets(predictions, [0, 1])
        self.assertEqual(len(targets), 1)
        self.assertEqual((targets[0]["class_id"], targets[0]["source_class_id"]), (0, 0))

    def test_overfit_gate_rejects_low_recall_or_extra_predictions(self):
        protocol = json.loads((ROOT / "configs/insplad_adapt_protocol.json").read_text())
        def records(predictions):
            return [dict(arm="overfit", elapsed_seconds=0.1, regions=1, peak_allocated_cuda_bytes=1,
                         metrics=operating_metrics(predictions, [[0, 0, 10, 10]], protocol)) for _ in range(2)]
        correct = [dict(box=[0, 0, 10, 10], score=0.9)]
        self.assertTrue(overfit_gate(records(correct), protocol)["passed"])
        self.assertFalse(overfit_gate(records([]), protocol)["passed"])
        self.assertFalse(overfit_gate(records(correct + [dict(box=[20, 20, 30, 30], score=0.9)]), protocol)["passed"])

    def test_saved_prediction_verifier_rejects_forged_metrics(self):
        protocol = dict(operating_confidences=[.05, .25], iou_thresholds=[.3, .5], nms_iou=.5)
        raw = [dict(class_id=0, score=.9, box=[0, 0, 10, 10], region=0)]
        targets = collapse_targets(raw, [0])
        rows = [dict(image_id=1, sha256="fixture", references=[dict(box=[0, 0, 10, 10])])]
        record = dict(arm="unit", image_id=1, image_sha256="fixture", prediction_file="prediction.json",
                      metrics=operating_metrics(targets, [[0, 0, 10, 10]], protocol))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prediction.json").write_text(json.dumps(dict(
                arm="unit", image_id=1, image_sha256="fixture", raw_predictions=raw,
                classwise_merged_predictions=raw, target_predictions=targets, target_ids=[0])))
            self.assertEqual(verify_records([record], rows, root, protocol), 1)
            record["metrics"]["conf_0.05_iou_0.50"]["tp"] = 2
            with self.assertRaisesRegex(ValueError, "metrics"):
                verify_records([record], rows, root, protocol)

    def test_control_addendum_preserves_frozen_base_protocol(self):
        import hashlib
        raw = (ROOT / "configs/insplad_adapt_protocol.json").read_bytes()
        control = json.loads((ROOT / "configs/insplad_adapt_control_v1.json").read_text())
        self.assertEqual(control["base_protocol_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertFalse(control["heldout_used_for_model_or_threshold_selection"])

    def test_report_rejects_changed_choices_or_a_trained_control(self):
        from copy import deepcopy
        from build_insplad_adaptation_report import verify_frozen_choices
        report = dict(selected_checkpoint_sha256="trained", untrained_control=dict(
            sha256="zero", optimizer_steps=0, saved_at="on_train_start"),
            control_addendum_sha256="addendum", dataset_manifest_sha256="dataset",
            prompt_selection=dict(selected_prompt="long_multi"))
        choices = dict(checkpoint_sha256="trained", untrained_control_sha256="zero",
            control_addendum_sha256="addendum", dataset_manifest_sha256="dataset",
            prompt_selection=report["prompt_selection"], protocol_sha256="protocol", selection_uses_holdout=False)
        verify_frozen_choices(report, choices, "protocol")
        for key, value in (("checkpoint_sha256", "other"), ("selection_uses_holdout", True)):
            with self.assertRaises(ValueError):
                verify_frozen_choices(report, {**choices, key: value}, "protocol")
        changed = deepcopy(report)
        changed["untrained_control"]["optimizer_steps"] = 1
        with self.assertRaisesRegex(ValueError, "optimizer"):
            verify_frozen_choices(changed, choices, "protocol")


if __name__ == "__main__":
    unittest.main()
