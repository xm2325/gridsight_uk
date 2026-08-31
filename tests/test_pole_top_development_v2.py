import unittest

from scripts.pole_top_development_v2 import derive_region


class PoleTopDevelopmentV2Tests(unittest.TestCase):
    rules = {
        "minimum_pole_height_over_width": 1.5,
        "maximum_crossarm_centre_relative_to_pole_top": .55,
        "crossarm_horizontal_padding_fraction": .15,
        "maximum_near_endpoint_distance_over_pole_height": .45,
        "minimum_far_to_near_distance_ratio": 1.5,
        "minimum_far_minus_near_over_pole_height": .15,
        "region_pole_width_multiplier": 1.5,
        "region_crossarm_extent_multiplier": 1.5,
        "region_minimum_image_fraction": .05,
        "ambiguity_margin_over_pole_height": .1,
    }

    def test_derives_unscored_region_from_unique_upper_association(self):
        result = derive_region([{"box": [45, 10, 55, 110]}],
                               [{"box": [20, 5, 80, 30]}], 200, 150, self.rules)
        self.assertEqual(result["status"], "geometry_candidate")
        self.assertIsNone(result["score"])
        self.assertFalse(result["physical_component_verified"])
        self.assertFalse(result["reference_truth"])

    def test_abstains_without_guarded_crossarm(self):
        result = derive_region([{"box": [45, 10, 55, 110]}], [], 200, 150, self.rules)
        self.assertEqual(result["status"], "unknown")

    def test_abstains_for_non_upright_pole(self):
        result = derive_region([{"box": [20, 20, 120, 50]}],
                               [{"box": [20, 15, 120, 35]}], 200, 150, self.rules)
        self.assertEqual(result["status"], "unknown")

    def test_abstains_for_ambiguous_two_pole_association(self):
        poles = [{"box": [45, 10, 55, 110]}, {"box": [50, 10, 60, 110]}]
        result = derive_region(poles, [{"box": [20, 5, 80, 30]}], 200, 150, self.rules)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("ambiguous", result["reason"])


if __name__ == "__main__":
    unittest.main()
