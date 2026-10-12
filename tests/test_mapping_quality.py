import unittest

from src.mapping_assist.analysis import FieldCandidate
from src.mapping_assist.matching import match_candidates


def candidate(label, section=None, kind="text", boxes=0):
    if kind == "checkbox" and isinstance(boxes, tuple):
        elements = tuple({"label": value, "bbox": (index * 12, 0, index * 12 + 10, 10)}
                         for index, value in enumerate(boxes))
    else:
        elements = tuple({"index": index, "bbox": (index * 12, 0, index * 12 + 10, 10)} for index in range(boxes))
    return FieldCandidate("field", label, None, (0, 0, 120, 30), kind, elements, section)


def custom(label="Circunstância do óbito", status="approved", data_type="text"):
    return {"key": "custom.death.circumstance", "label": label, "description": "Campo aprovado pelo revisor",
            "data_type": data_type, "generator": {"type": "unavailable"}, "status": status}


class MappingQualityTests(unittest.TestCase):
    def test_generic_date_requires_unique_key_in_section_and_compatible_boxes(self):
        match = match_candidates([candidate("Data", "I", "digit_sequence", 8)], "dnv")[0]
        self.assertEqual((match.status, match.key, match.method), ("matched", "birth.date", "rules"))
        self.assertNotEqual(match_candidates([candidate("Data", "I", "digit_sequence", 8)], "DO")[0].status, "matched")
        self.assertNotEqual(match_candidates([candidate("Data", "I", "digit_sequence", 7)], "dnv")[0].status, "matched")

    def test_approved_custom_definition_reused_conservatively(self):
        exact = match_candidates([candidate("Circunstância do óbito")], "DO", custom_definitions=[custom()])[0]
        self.assertEqual((exact.status, exact.key, exact.method), ("matched", "custom.death.circumstance", "custom"))
        self.assertNotEqual(match_candidates([candidate("Tipo de óbito")], "DO", custom_definitions=[custom()])[0].key,
                            exact.key)
        self.assertNotEqual(match_candidates([candidate("Circunstância do óbito")], "DO",
                                              custom_definitions=[custom(status="draft")])[0].key, exact.key)
        self.assertNotEqual(match_candidates([candidate("Circunstância do óbito", "I", "digit_sequence", 8)], "DO",
                                              custom_definitions=[custom()])[0].key, exact.key)

    def test_registry_match_has_precedence_over_custom_label(self):
        definition = custom(label="Nome do falecido")
        match = match_candidates([candidate("Nome do Falecido", "I")], "DO", custom_definitions=[definition])[0]
        self.assertEqual((match.key, match.method), ("deceased.name", "rules"))

    def test_custom_checkbox_requires_known_choices(self):
        definition = custom(label="Tipo de óbito")
        definition["generator"] = {"type": "choices", "values": ["Fetal", "Não fetal"]}
        matched = match_candidates([candidate("Tipo de óbito", "I", "checkbox", ("Fetal", "Não fetal"))], "DO",
                                   custom_definitions=[definition])[0]
        self.assertEqual((matched.status, matched.key), ("matched", definition["key"]))
        unmatched = match_candidates([candidate("Tipo de óbito", "I", "checkbox", ("Fetal", "Ignorado"))], "DO",
                                     custom_definitions=[definition])[0]
        self.assertNotEqual(unmatched.key, definition["key"])


if __name__ == "__main__":
    unittest.main()
