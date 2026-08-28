"""Geometry, matching, source-identity and holdout guards; no GPU required."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("roihu_demo", Path(__file__).resolve().parents[1] / "scripts/roihu_demo_ablation.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_insplad100 as insplad
import roihu_benchmark100 as benchmark


class RoihuDemoTests(unittest.TestCase):
    def test_holdout_or_unknown_source_is_rejected(self):
        for rid in ("POS_3437435", "POS_7561805", "POS_7630781", "unknown"):
            with self.subTest(rid=rid), self.assertRaises(ValueError):
                demo.select_sources([rid])

    def test_duplicate_sources_cannot_inflate_sample_size(self):
        with self.assertRaises(ValueError):
            demo.select_sources(["POS_2326530", "POS_2326530"])

    def test_cached_derivative_cannot_impersonate_exact_original(self):
        row = demo.select_sources(["POS_2326530"])[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / row["filename"]
            path.write_bytes(b"different image bytes")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                demo.verify_source(path, row)

    def test_original_coordinate_labels_remain_scaled_to_source(self):
        row = demo.select_sources(["POS_2326530"])[0]
        references, _ = demo.insulator_references(row)
        self.assertEqual(len(references), 6)
        self.assertGreater(min(b[2] - b[0] for b in references), 100)

    def test_odd_sized_tiling_covers_every_axis_without_duplicates(self):
        for length in (19, 1280, 1281, 3168, 4752):
            starts = demo.axis_starts(length, 1280, 0.25)
            self.assertEqual(len(starts), len(set(starts)))
            covered = set()
            for start in starts:
                covered.update(range(start, min(start + 1280, length)))
            self.assertEqual(covered, set(range(length)))

    def test_invalid_overlap_is_rejected(self):
        for overlap in (-0.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                demo.windows(100, 100, overlap=overlap)

    def test_tile_boxes_are_translated_then_clipped(self):
        self.assertEqual(demo.offset_box([-5, -5, 200, 300], 90, 80, 150, 160), [85, 75, 150, 160])

    def test_nms_suppresses_same_class_only(self):
        boxes = [{"box": [0, 0, 10, 10], "score": s, "class_id": c} for s, c in ((0.9, 2), (0.8, 2), (0.7, 0))]
        self.assertEqual([(p["score"], p["class_id"]) for p in demo.nms(boxes)], [(0.9, 2), (0.7, 0)])

    def test_one_reference_can_match_only_once(self):
        predictions = [{"box": [0, 0, 10, 10], "score": s} for s in (0.9, 0.8)]
        metrics = demo.evaluate(predictions, [[0, 0, 10, 10], [20, 20, 30, 30]], 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))

    def test_empty_predictions_preserve_false_negatives(self):
        metrics = demo.evaluate([], [[0, 0, 10, 10]], 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["recall"]), (0, 0, 1, 0))

    def test_coco_mapping_does_not_count_insulator_shackles(self):
        annotations = [dict(id=c, category_id=c, bbox=[10, 20, 30, 40], iscrowd=0)
                       for c in (7, 8, 11, 12, 13, 14, 15, 16)]
        refs = insplad.target_references(annotations)
        self.assertEqual([r["category_id"] for r in refs], [7, 8])
        self.assertEqual(refs[0]["box"], [10, 20, 40, 60])

    def test_capture_sampling_is_stable_and_balances_prefixes(self):
        rows = [dict(id=10 * g + i, file_name=f"{g}-1_DJI_{i:04d}.jpg")
                for g in range(3) for i in range(4)]
        ordered = list(insplad.ordered_candidates(rows))
        reversed_order = list(insplad.ordered_candidates(list(reversed(rows))))
        self.assertEqual(ordered, reversed_order)
        self.assertEqual(len({insplad.capture_group(r["file_name"]) for r in ordered[:3]}), 3)
        self.assertEqual(len({r["id"] for r in ordered}), len(rows))

    def test_crowds_and_unsafe_source_paths_fail_closed(self):
        with self.assertRaises(ValueError):
            insplad.capture_group("../1-1_DJI_0001.jpg")
        with self.assertRaises(ValueError):
            insplad.target_references([dict(id=1, category_id=7, bbox=[0, 0, 10, 10], iscrowd=1)])

    def test_prespecified_confidence_points_are_evaluated_separately(self):
        predictions = [dict(box=[0, 0, 10, 10], score=s) for s in (0.8, 0.1)]
        protocol = dict(operating_confidences=[0.05, 0.25], iou_thresholds=[0.5])
        metrics = benchmark.operating_metrics(predictions, [[0, 0, 10, 10]], protocol)
        self.assertEqual(metrics["conf_0.05_iou_0.50"]["fp"], 1)
        self.assertEqual(metrics["conf_0.25_iou_0.50"]["fp"], 0)

    def test_summary_pools_counts_not_per_image_ratios(self):
        key = "conf_0.05_iou_0.50"
        rows = [dict(arm="n640", metrics={key: demo.counts_to_metrics(*counts)},
                     elapsed_seconds=0.5, regions=1, peak_allocated_cuda_bytes=100)
                for counts in ((3, 1, 0), (0, 0, 2))]
        result = benchmark.summarize(rows, ["n640"], dict(operating_confidences=[0.05], iou_thresholds=[0.5]))
        metrics = result["n640"]["metrics"][key]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (3, 1, 2))
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.6)


if __name__ == "__main__":
    unittest.main()
