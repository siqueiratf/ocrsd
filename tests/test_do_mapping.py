import json
import random
import tempfile
import unittest
from datetime import date
from pathlib import Path

import fitz

from src.semantic.constraints import completed_years
from src.semantic.plan import semantic_plan
from src.synthetic_dataset import generate_document


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/do/do.pdf"
MAPPING = ROOT / "documents/do/mappings/mapeamento_do_v1.json"


class DOModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    def test_official_reference_geometry_and_generation_scope(self):
        with fitz.open(TEMPLATE) as pdf:
            self.assertEqual(len(pdf), 1)
            self.assertEqual(tuple(pdf[0].rect[2:]), (612, 792))
        self.assertIn("Apêndice A", self.mapping["reference_manual"])
        self.assertIn("Reprodução", self.mapping["template_provenance"])
        plan = semantic_plan(self.mapping["fields"], "declaracao_obito")
        self.assertFalse(plan.issues)
        ids = [field["field_id"] for field in self.mapping["fields"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 103)
        unsupported = {field["field_id"] for field in self.mapping["fields"] if field.get("unsupported_reason")}
        self.assertTrue({"9_meses", "9_dias", "9_horas", "9_minutos", "40_a_causa", "40_a_cid",
                         "47", "48"} <= unsupported)
        self.assertEqual(len(next(field for field in self.mapping["fields"]
                                  if field["field_id"] == "9_ignorado")["elements"]), 1)
        for field in self.mapping["fields"]:
            x1, y1, x2, y2 = field["bbox"]
            self.assertTrue(0 <= x1 < x2 <= 2550 and 0 <= y1 < y2 <= 3300, field["field_id"])
            for element in field.get("elements", []):
                ex1, ey1, ex2, ey2 = element["bbox"]
                self.assertTrue(0 <= ex1 < ex2 <= 2550 and 0 <= ey1 < ey2 <= 3300, field["field_id"])
            if field["field_id"] not in plan.bindings:
                self.assertTrue(field.get("unsupported_reason"), field["field_id"])

    def test_supported_do_fields_generate_consistent_records(self):
        selected = {field["field_id"] for field in self.mapping["fields"] if not field.get("unsupported_reason")}
        with tempfile.TemporaryDirectory() as temp:
            for seed in range(10):
                output = generate_document(TEMPLATE, self.mapping, selected, Path(temp), str(seed),
                                           random.Random(seed), handwriting=False, text_handwriting=False)
                with fitz.open(output.pdf_path) as pdf:
                    self.assertEqual(len(pdf), 1)
                truth = json.loads(output.json_path.read_text(encoding="utf-8"))
                values = {field["field_id"]: field["value"] for field in truth["fields"]}
                record = truth["semantic_record"]
                born = date.fromisoformat(record["deceased.birth_date"])
                died = date.fromisoformat(record["death.date"])
                registered = date.fromisoformat(record["death_registration.date"])
                self.assertLessEqual(born, died)
                self.assertLessEqual(died, registered)
                self.assertEqual(record["deceased.age"], completed_years(born, died))
                self.assertEqual(values["2"], died.strftime("%d%m%Y"))
                self.assertEqual(values["8"], born.strftime("%d%m%Y"))
                self.assertEqual(values["46"], died.strftime("%d%m%Y"))
                self.assertEqual(values["55"], registered.strftime("%d%m%Y"))
                self.assertEqual(values["18"], record["deceased.residence.municipality"])
                self.assertEqual(values["19"], record["deceased.residence.uf"])
                self.assertEqual(values["25"], record["death.place.municipality"])
                self.assertEqual(values["26"], record["death.place.uf"])
                self.assertEqual(values["1"], "2")
                self.assertEqual(values["20"], "1")
                self.assertEqual(values["38"], "1")
                self.assertEqual(values["39"], "2")
                self.assertEqual(values["43"], "1")
                self.assertGreaterEqual(record["deceased.age"], 1)
                self.assertFalse(any(field_id.startswith("40_") for field_id in values))


if __name__ == "__main__":
    unittest.main()
