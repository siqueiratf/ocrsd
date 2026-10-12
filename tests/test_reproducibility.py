import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import fitz

from src.synthetic_dataset import generate_dataset, generate_document


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.template = self.root / "template.pdf"
        with fitz.open() as pdf:
            pdf.new_page(width=400, height=300)
            pdf.save(self.template)
        self.mapping = self.root / "mapping.json"
        self.definition = {
            "template_id": "certidao_nascimento", "version": 7, "dpi": 72, "page_size_px": [400, 300],
            "shared_rules": {"date": {"format": "DD-MM-AAAA"}},
            "fields": [
                {"field_id": "birth", "label": "Data de nascimento", "data_type": "date",
                 "semantic": "birth.date", "rule_refs": ["date"], "bbox": [10, 20, 300, 50]},
                {"field_id": "duplicate", "label": "Nascimento", "data_type": "date",
                 "semantic": "birth.date", "rule_refs": ["date"], "bbox": [10, 70, 300, 100]},
                {"field_id": "unused", "label": "Nome", "data_type": "person_name", "bbox": [10, 120, 300, 150]},
            ],
        }
        self.mapping.write_text(json.dumps(self.definition), encoding="utf-8")

    def generate(self, output, seed=None):
        return generate_dataset(self.template, self.mapping, {"birth", "duplicate"}, self.root / output, 2,
                                seed=seed, handwriting=False, text_handwriting=False)

    def test_unseeded_artifacts_reproduce_after_source_mapping_changes(self):
        first = self.generate("first")
        manifest = json.loads((self.root / "first/dataset.json").read_text())
        seed = manifest["seed"]
        self.assertIsInstance(seed, int)
        self.assertTrue(0 <= seed < 2**32)
        self.assertIsNone(manifest["requested_seed"])
        reproduction = manifest["reproduction"]
        self.assertEqual(reproduction["mapping_version"], 7)
        self.assertEqual(reproduction["template_sha256"], hashlib.sha256(self.template.read_bytes()).hexdigest())
        snapshot = reproduction["mapping_snapshot"]
        self.assertEqual(snapshot["fields"][0]["validation"]["format"], "DD-MM-AAAA")
        encoded = (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        self.assertEqual(reproduction["mapping_sha256"], hashlib.sha256(encoded).hexdigest())
        self.mapping.write_text('{"fields": []}', encoding="utf-8")
        restored = self.root / "restored.json"
        restored.write_text(json.dumps(snapshot), encoding="utf-8")
        second = generate_dataset(self.template, restored, set(reproduction["selected_ids"]), self.root / "second",
                                  manifest["count"], seed=seed, **reproduction["options"])
        for index, (before, after) in enumerate(zip(first, second), 1):
            original = json.loads(before.json_path.read_text())
            regenerated = json.loads(after.json_path.read_text())
            self.assertEqual(original, regenerated)
            self.assertEqual(original["generation"]["seed"], seed)
            self.assertEqual(original["generation"]["index"], index)
            self.assertIsInstance(original["semantic_seed"], int)
            self.assertEqual(original["semantic_context"], "certidao_nascimento")
            self.assertEqual(before.values, after.values)
            with fitz.open(before.pdf_path) as a, fitz.open(after.pdf_path) as b:
                self.assertEqual(a[0].get_pixmap().samples, b[0].get_pixmap().samples)

    def test_explicit_seed_preserves_document_rng_stream(self):
        generated = self.generate("dataset", seed=35)
        rng = random.Random(35)
        for index, expected in enumerate(generated, 1):
            direct = generate_document(self.template, self.definition, {"birth", "duplicate"}, self.root / "direct",
                                       f"synthetic_{index:04d}", rng, handwriting=False, text_handwriting=False)
            expected_truth = json.loads(expected.json_path.read_text())
            direct_truth = json.loads(direct.json_path.read_text())
            self.assertEqual(expected_truth["fields"], direct_truth["fields"])
            self.assertEqual(expected_truth["semantic_seed"], direct_truth["semantic_seed"])
            self.assertEqual(expected.values, direct.values)
            saved_state = direct_truth["generation"]["random_state"]
            resumed_rng = random.Random()
            resumed_rng.setstate((saved_state[0], tuple(saved_state[1]), saved_state[2]))
            replay = generate_document(self.template, direct_truth["generation"]["mapping_snapshot"],
                                       set(direct_truth["generation"]["selected_ids"]), self.root / "replay",
                                       f"synthetic_{index:04d}", resumed_rng, **direct_truth["generation"]["options"])
            self.assertEqual(direct.values, replay.values)


if __name__ == "__main__":
    unittest.main()
