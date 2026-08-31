import unittest

from scripts.roihu_crossarm_grounding_v1 import pole_associated, select_arm, suppress_poles


class CrossarmGroundingV1Tests(unittest.TestCase):
    rules = {
        "pole_centre_horizontal_candidate_padding": .15,
        "candidate_centre_min_relative_to_pole_top": -.25,
        "candidate_centre_max_relative_to_pole_top": .55,
        "minimum_candidate_width_over_pole_width": .5,
        "maximum_candidate_height_over_pole_height": .8,
    }

    def test_min_area_containment_suppresses_nested_pole(self):
        rows = [{"class_id": 0, "score": .9, "box": [0, 0, 10, 100]},
                {"class_id": 0, "score": .2, "box": [1, 20, 9, 90]},
                {"class_id": 1, "score": .9, "box": [0, 0, 10, 10]}]
        result = suppress_poles(rows, .05, .5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], .9)

    def test_pole_association_requires_upper_support_geometry(self):
        poles = [{"box": [45, 20, 55, 100]}]
        self.assertTrue(pole_associated({"box": [10, 10, 90, 30]}, poles, self.rules))
        self.assertFalse(pole_associated({"box": [10, 80, 90, 95]}, poles, self.rules))

    def test_selection_uses_epri_f1(self):
        records = [{"predictions": [[{"class_id": 1, "score": .3, "box": [0, 0, 10, 10]}]],
                    "poles": [{"box": [4, 0, 6, 20]}],
                    "references": [{"class_id": 1, "box": [0, 0, 10, 10]}]}]
        result = select_arm(records, ["prompt"], ["raw", "pole_associated"], [.1, .2], self.rules)
        self.assertEqual(result["selected"]["threshold"], .2)
        self.assertEqual(result["selected"]["variant"], "pole_associated")


if __name__ == "__main__":
    unittest.main()
