import unittest

from scripts.evaluate_mapping import pairs, score


class MappingEvaluationTests(unittest.TestCase):
    def test_assignment_is_greedy_and_one_to_one(self):
        proposed = [{"bbox": [0, 0, 100, 100]}, {"bbox": [5, 5, 95, 95]}]
        reference = [{"bbox": [0, 0, 100, 100]}, {"bbox": [200, 0, 300, 100]}]
        self.assertEqual(pairs(proposed, reference, 0.3), [(0, 0, 1.0)])

    def test_detection_semantics_elements_and_corrections(self):
        boxes = [{"element_id": f"e{index}", "bbox": [index * 20, 300, index * 20 + 15, 315]} for index in range(4)]
        proposal = {"page_size_px": [400, 400], "fields": [
            {"field_id": "a", "bbox": [0, 0, 100, 100], "semantic": "newborn.name", "suggestion": {"match": "matched"}},
            {"field_id": "b", "bbox": [120, 0, 220, 100], "semantic": "newborn.sex", "suggestion": {"match": "matched"}},
            {"field_id": "c", "bbox": [0, 290, 100, 320], "elements": boxes[:3], "suggestion": {"match": "new"}},
            {"field_id": "extra", "bbox": [0, 150, 100, 250], "suggestion": {"match": "ambiguous"}},
        ]}
        reference = {"page_size_px": [400, 400], "fields": [
            {"field_id": "name", "label": "Nome", "bbox": [0, 0, 100, 100], "semantic": "newborn.name"},
            {"field_id": "cpf", "label": "CPF", "bbox": [120, 0, 220, 100], "semantic": "newborn.cpf"},
            {"field_id": "digits", "label": "Caixas", "bbox": [0, 290, 100, 320], "elements": boxes},
            {"field_id": "missing", "label": "Data", "bbox": [120, 150, 220, 250], "semantic": "birth.date"},
        ]}
        metrics = score(proposal, reference, "dnv", 0.3)
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["matched_fields"]), (0.75, 0.75, 3))
        self.assertEqual(metrics["mean_iou_matched"], 1.0)
        self.assertEqual(metrics["element_recall"], 0.75)
        self.assertEqual((metrics["semantic_compared"], metrics["semantic_accuracy"]), (2, 0.5))
        self.assertEqual(metrics["matched_status_counts"], {"matched": 2, "new": 1})
        self.assertEqual(metrics["unmatched_reference"], ["missing"])
        self.assertEqual(metrics["false_positives"], ["extra"])
        self.assertEqual(metrics["manual_corrections"], 3)

    def test_page_dimensions_must_match(self):
        with self.assertRaisesRegex(ValueError, "Dimensões diferentes"):
            score({"page_size_px": [100, 100], "fields": []}, {"page_size_px": [200, 100], "fields": []}, "dnv", 0.3)


if __name__ == "__main__":
    unittest.main()
