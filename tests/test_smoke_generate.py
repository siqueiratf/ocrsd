import copy
import unittest

from scripts.smoke_generate import expected_content


class SmokeGroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.truth = {"semantic_record": {"guardian.name": None}, "fields": [
            {"field_id": "nome", "value": "Ana Silva"},
            {"field_id": "responsavel", "value": "", "blank": True,
             "semantic": {"key": "guardian.name", "value": None}},
        ]}

    def test_only_optional_semantic_absence_exempts_rendering(self):
        self.assertEqual(expected_content(self.truth), {"nome"})

    def test_empty_required_or_unexplained_fields_fail(self):
        for field in ({"field_id": "nome", "value": ""},
                      {"field_id": "nome", "value": "", "blank": True},
                      {"field_id": "nome", "value": "", "blank": True,
                       "semantic": {"key": "newborn.name", "value": None}},
                      {"field_id": "nome", "value": None},
                      {"field_id": "nome", "value": "Ana Silva", "blank": True}):
            with self.subTest(field=field), self.assertRaises(AssertionError):
                expected_content({"semantic_record": {"newborn.name": None}, "fields": [field]})

    def test_blank_must_agree_with_record_and_annotation(self):
        for snapshot in ({}, {"guardian.name": "Ana Silva"}):
            truth = {**self.truth, "semantic_record": snapshot}
            with self.subTest(snapshot=snapshot), self.assertRaises(AssertionError):
                expected_content(truth)
        for changes in ({"blank": False}, {"semantic": {"key": "guardian.name"}},
                        {"semantic": {"key": "guardian.name", "value": "Ana Silva"}}):
            truth = copy.deepcopy(self.truth)
            truth["fields"][1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                expected_content(truth)

    def test_conditional_blank_matches_record(self):
        field = {"field_id": "local_externo", "value": "", "blank": True,
                 "render_condition": {"key": "birth.occurrence_type", "values": ["Domicílio", "Outros"]}}
        truth = {"semantic_record": {"birth.occurrence_type": "Hospital"}, "fields": [field]}
        self.assertEqual(expected_content(truth), set())
        truth["semantic_record"]["birth.occurrence_type"] = "Domicílio"
        with self.assertRaises(AssertionError):
            expected_content(truth)


if __name__ == "__main__":
    unittest.main()
