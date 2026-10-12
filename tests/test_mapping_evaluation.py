import unittest

from scripts.evaluate_mapping import _pairs, score_proposal


class MappingEvaluationTests(unittest.TestCase):
    def test_assignment_does_not_reuse_a_reference_region(self):
        proposed = [{"bbox": [0, 0, 100, 100]}, {"bbox": [5, 5, 95, 95]}]
        reference = [{"bbox": [0, 0, 100, 100]}, {"bbox": [200, 0, 300, 100]}]
        self.assertEqual(len(_pairs(proposed, reference)), 1)

    def test_detection_and_classification_have_separate_denominators(self):
        proposal = {"page_size_px": [300, 300], "fields": [
            {"field_id": "a", "bbox": [0, 0, 100, 100], "semantic": "newborn.name"},
            {"field_id": "b", "bbox": [120, 0, 220, 100], "semantic": "newborn.sex"},
            {"field_id": "extra", "bbox": [0, 150, 100, 250]},
        ]}
        reference = {"page_size_px": [300, 300], "fields": [
            {"field_id": "name", "bbox": [0, 0, 100, 100], "semantic": "newborn.name"},
            {"field_id": "sex", "bbox": [120, 0, 220, 100], "semantic": "newborn.cpf"},
            {"field_id": "missing", "bbox": [120, 150, 220, 250], "semantic": "birth.date"},
        ]}
        metrics = score_proposal(proposal, reference, "dnv")
        self.assertEqual((metrics["detection_precision"], metrics["detection_recall"]), (0.6667, 0.6667))
        self.assertEqual(metrics["mean_iou_matched"], 1.0)
        self.assertEqual(metrics["semantic_accuracy_on_matched_regions"], 0.5)
        self.assertEqual(metrics["semantic_correct_over_reference"], 0.3333)
        self.assertEqual(metrics["region_misses_or_semantic_errors"], 2)
        self.assertEqual(metrics["suggestions_requiring_human_review"], 3)
        self.assertEqual(metrics["unmatched_reference"], ["missing"])
        self.assertEqual(metrics["unmatched_proposals"], ["extra"])

    def test_page_dimensions_must_match(self):
        with self.assertRaisesRegex(ValueError, "Dimensões diferentes"):
            score_proposal({"page_size_px": [100, 100], "fields": []},
                           {"page_size_px": [200, 100], "fields": []}, "dnv")


if __name__ == "__main__":
    unittest.main()
