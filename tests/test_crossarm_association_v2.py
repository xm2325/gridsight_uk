import unittest

from scripts.crossarm_association_v2 import guard, upright_pole_associated


class CrossarmAssociationV2Tests(unittest.TestCase):
    rules = {
        "pole_centre_horizontal_candidate_padding": .15,
        "candidate_centre_min_relative_to_pole_top": -.25,
        "candidate_centre_max_relative_to_pole_top": .55,
        "minimum_candidate_width_over_pole_width": .5,
        "maximum_candidate_height_over_pole_height": .45,
        "minimum_pole_height_over_width": 1.5,
        "maximum_candidate_centre_offset_over_pole_width": 1.5,
    }

    def test_rejects_horizontal_pole_input(self):
        candidate = {"score": .4, "box": [0, 0, 100, 20]}
        poles = [{"box": [20, 0, 90, 30]}]
        self.assertFalse(upright_pole_associated(candidate, poles, self.rules))

    def test_rejects_pole_shaft_candidate(self):
        candidate = {"score": .4, "box": [45, 10, 60, 90]}
        poles = [{"box": [45, 0, 60, 100]}]
        self.assertFalse(upright_pole_associated(candidate, poles, self.rules))

    def test_keeps_compact_upper_support_candidate(self):
        candidate = {"score": .4, "box": [20, 5, 80, 30]}
        poles = [{"box": [45, 0, 60, 100]}]
        self.assertTrue(upright_pole_associated(candidate, poles, self.rules))

    def test_guard_applies_raw_threshold(self):
        poles = [{"box": [45, 0, 60, 100]}]
        predictions = [{"score": .29, "box": [20, 5, 80, 30]},
                       {"score": .31, "box": [20, 5, 80, 30]}]
        self.assertEqual(guard(predictions, poles, self.rules, .3), [predictions[1]])


if __name__ == "__main__":
    unittest.main()
