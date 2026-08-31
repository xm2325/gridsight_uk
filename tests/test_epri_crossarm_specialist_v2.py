import unittest

from scripts.prepare_epri_crossarm_specialist_v2 import label_text
from scripts.roihu_epri_crossarm_specialist_v2 import crossarm_references, select_threshold
from scripts.build_crossarm_specialist_v2_report import RESULT_SHA


class CrossarmSpecialistV2Tests(unittest.TestCase):
    def test_labels_keep_only_crossarm_and_remap_to_zero(self):
        record = {"width": 100, "height": 50, "references": [
            {"class_name": "pole", "box": [0, 0, 10, 50]},
            {"class_name": "crossarm", "box": [10, 5, 90, 15]},
        ]}
        self.assertEqual(label_text(record), "0 0.500000000 0.200000000 0.800000000 0.200000000\n")

    def test_crossarm_reference_mapping_is_non_mutating(self):
        pole = {"class_name": "pole", "class_id": 0, "box": [0, 0, 1, 1]}
        arm = {"class_name": "crossarm", "class_id": 1, "box": [0, 0, 1, 1]}
        result = crossarm_references({"references": [pole, arm]})
        self.assertEqual(result[0]["class_id"], 0)
        self.assertEqual(arm["class_id"], 1)

    def test_threshold_uses_development_f1_and_higher_tie(self):
        records = [{"predictions": [{"class_id": 0, "score": .2, "box": [0, 0, 10, 10]}],
                    "references": [{"class_id": 0, "box": [0, 0, 10, 10]}]}]
        selected = select_threshold(records, [.05, .1, .25])
        self.assertEqual(selected["selected"]["threshold"], .1)
        self.assertEqual(selected["selected"]["f1"], 1.0)

    def test_report_pins_real_result(self):
        self.assertEqual(RESULT_SHA, "6561ce49bfe8cae93c2897bae049e870914b6e2458b68dd0edf7c0a830fa148f")


if __name__ == "__main__":
    unittest.main()
