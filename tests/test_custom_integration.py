import json
import random
import tempfile
import unittest
from pathlib import Path

import fitz

from src.synthetic_dataset import field_texts, generate_dataset
from src.web.services import ServiceError, mapping_issues, validate_mapping


class CustomSemanticIntegrationTests(unittest.TestCase):
    def test_approved_definition_reuses_one_value_per_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=500, height=300)
                pdf.save(template)
            spec = {"key": "custom.circunstancia", "label": "Circunstância", "description": "Valor controlado",
                    "data_type": "text", "generator": {"type": "choices", "values": ["Opção A", "Opção B"]}}
            definition = {"template_id": "modelo_web", "dpi": 72, "page_size_px": [500, 300],
                          "semantic_definitions": {spec["key"]: spec}, "fields": [
                              {"field_id": name, "label": name, "data_type": "text", "semantic": spec["key"], "bbox": box}
                              for name, box in (("primeiro", [20, 20, 200, 50]), ("repetido", [20, 70, 200, 100]))]}
            validated = validate_mapping(definition)
            self.assertEqual(mapping_issues(validated), [])
            self.assertIn(field_texts(validated["fields"][0], random.Random(1),
                                      {"semantic_definitions": validated["semantic_definitions"]})[0],
                          spec["generator"]["values"])
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps(validated), encoding="utf-8")
            docs = generate_dataset(template, mapping, {"primeiro", "repetido"}, root / "dataset", 3,
                                    seed=123, handwriting=False, text_handwriting=False)
            for doc in docs:
                truth = json.loads(doc.json_path.read_text())
                first, second = truth["fields"]
                self.assertEqual(first["value"], second["value"])
                self.assertEqual(first["semantic"]["key"], spec["key"])
                self.assertEqual(first["semantic"]["source"], "custom")
                self.assertEqual(truth["semantic_record"][spec["key"]], first["value"])

    def test_unconfigured_medical_definition_blocks_generation(self):
        spec = {"key": "custom.death_cause", "label": "Causa da morte", "description": "CID validado",
                "data_type": "text", "generator": {"type": "unavailable"}}
        definition = {"template_id": "modelo_web", "dpi": 72, "page_size_px": [500, 300],
                      "semantic_definitions": {spec["key"]: spec}, "fields": [
                          {"field_id": "causa", "label": "Causa", "data_type": "text", "semantic": spec["key"],
                           "bbox": [20, 20, 200, 50]}]}
        self.assertIn("configure um gerador validado", mapping_issues(validate_mapping(definition))[0])
        definition["semantic_definitions"].clear()
        with self.assertRaises(ServiceError):
            validate_mapping(definition)


if __name__ == "__main__":
    unittest.main()
