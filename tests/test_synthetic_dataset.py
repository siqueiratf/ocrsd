import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from src.synthetic_dataset import (
    _apply_ballpoint_texture, apply_shared_rules, generate_dataset, is_numeric_text,
    random_handwriting_style, random_value, render_checkbox_mark,
    render_handwritten_numeric, render_handwritten_text, render_neural_text,
)
from src.neural_handwriting import NeuralPen


class SyntheticValueTests(unittest.TestCase):
    def test_respects_allowed_values(self):
        field = {"data_type": "enum", "validation": {"allowed_values": ["A", "B"]}}
        self.assertIn(random_value(field, random.Random(1)), {"A", "B"})

    def test_generates_typed_values(self):
        rng = random.Random(3)
        self.assertRegex(random_value({"data_type": "date"}, rng), r"^\d{2}/\d{2}/\d{4}$")
        self.assertRegex(random_value({"data_type": "cep"}, rng), r"^\d{5}-\d{3}$")
        self.assertEqual(len(random_value({"data_type": "cns"}, rng)), 15)

    def test_applies_shared_rules_to_digit_sequences(self):
        field_map = apply_shared_rules({
            "shared_rules": {
                "RULE_APGAR_RANGE": {"min": 0, "max": 10},
                "RULE_CEP": {"digit_count": 8, "pattern": "^[0-9]{8}$"},
            },
            "fields": [
                {"field_id": "apgar", "description": "Indice de Apgar", "data_type": "digit_sequence",
                 "rule_refs": ["RULE_APGAR_RANGE"], "elements": [{"index": 0}, {"index": 1}]},
                {"field_id": "cep", "description": "CEP", "data_type": "digit_sequence",
                 "rule_refs": ["RULE_CEP"], "elements": [{"index": index} for index in range(8)]},
            ],
        })

        rng = random.Random(5)
        apgar = random_value(field_map["fields"][0], rng, {})
        cep = random_value(field_map["fields"][1], rng, {})

        self.assertLessEqual(int(apgar), 10)
        self.assertRegex(apgar, r"^\d{2}$")
        self.assertRegex(cep, r"^\d{8}$")

    def test_generates_coherent_location_fields(self):
        context: dict = {}
        rng = random.Random(8)

        cep = random_value({"field_id": "10", "description": "CEP (endereco da ocorrencia)",
                            "data_type": "digit_sequence", "elements": [{"index": index} for index in range(8)]}, rng, context)
        city = random_value({"field_id": "12", "description": "Municipio de ocorrencia"}, rng, context)
        uf = random_value({"field_id": "13", "description": "UF (local da ocorrencia)", "data_type": "uf"}, rng, context)

        profile = context["locations"]["occurrence"]
        self.assertEqual(cep, profile["cep"])
        self.assertEqual(city, profile["city"])
        self.assertEqual(uf, profile["uf"])

    def test_generates_plausible_gestational_history_counts(self):
        context: dict = {}
        rng = random.Random(11)

        gestacoes = int(random_value({"field_id": "30_gestacoes_anteriores",
                                      "description": "N gestacoes anteriores",
                                      "data_type": "digit_sequence",
                                      "elements": [{"index": 0}, {"index": 1}]}, rng, context))
        vivos = int(random_value({"field_id": "30_nascidos_vivos",
                                  "description": "N nascidos vivos",
                                  "data_type": "digit_sequence",
                                  "elements": [{"index": 0}, {"index": 1}]}, rng, context))

        self.assertLessEqual(gestacoes, 8)
        self.assertLessEqual(vivos, gestacoes)

    def test_detects_values_supported_by_numeric_handwriting(self):
        self.assertTrue(is_numeric_text("12/08/2026 14:35"))
        self.assertTrue(is_numeric_text("3,250"))
        self.assertFalse(is_numeric_text("ABC123"))

    def test_renders_deterministic_transparent_handwriting(self):
        first_rng = random.Random(9)
        first_style = random_handwriting_style(first_rng)
        first = render_handwritten_numeric("58001", 180, 40, first_rng, first_style)
        second_rng = random.Random(9)
        second_style = random_handwriting_style(second_rng)
        second = render_handwritten_numeric("58001", 180, 40, second_rng, second_style)

        self.assertEqual(first.mode, "RGBA")
        self.assertEqual(first.size, (180, 40))
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertGreater(first.getchannel("A").getbbox()[2], 0)

    def test_ballpoint_texture_creates_weak_ink_regions(self):
        solid_stroke = np.full((32, 96), 220, dtype=np.uint8)

        textured = _apply_ballpoint_texture(solid_stroke, random.Random(4), 1.0)

        self.assertLess(int(textured.min()), 120)
        self.assertGreater(int(textured.max()), 180)
        self.assertGreater(float(textured.std()), 20)

    def test_renders_text_with_graphite_colored_strokes(self):
        image = render_handwritten_text(
            "Ana Beatriz", 260, 48, random.Random(12), intensity=0.7, medium="pencil",
        )

        alpha = np.asarray(image.getchannel("A"))
        rgb = np.asarray(image)[alpha > 30, :3]
        self.assertGreater(alpha.max(), 0)
        self.assertLess(abs(float(rgb[:, 0].mean()) - float(rgb[:, 2].mean())), 12)

    def test_renders_checkbox_mark_as_handwriting(self):
        image, metadata = render_checkbox_mark(
            48, 48, random.Random(19), style=random_handwriting_style(random.Random(19)),
        )

        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (48, 48))
        self.assertGreater(np.asarray(image.getchannel("A")).sum(), 0)
        self.assertIn(metadata["rendering"]["mark_variant"], {"middle_stroke", "upper_stroke", "vertical_stroke", "x"})

    def test_neural_render_retries_transient_model_failure(self):
        model = unittest.mock.Mock()
        points = np.array([[0, 0, 0], [10, 1, 0], [20, 0, 1]], dtype=float)
        model.sample.side_effect = [ValueError("falha transitória"), points]

        image, metadata = render_neural_text(
            "15092026", 220, 48, NeuralPen(model="sjvasquez", writer=1), model, random.Random(7),
        )

        self.assertEqual(image.size, (220, 48))
        self.assertEqual(metadata["backend"], "sjvasquez")
        self.assertEqual(model.sample.call_count, 2)


class SyntheticDatasetTests(unittest.TestCase):
    def test_printed_text_fits_short_certificate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page(width=300, height=300)
                document.save(template)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "short", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "cpf", "bbox": [20, 20, 1200, 74],
                    "render_as_printed": True,
                    "validation": {"allowed_values": ["123.456.789-09"]},
                }],
            }), encoding="utf-8")

            generated = generate_dataset(template, mapping, {"cpf"}, root / "out", 1, seed=42)

            with fitz.open(generated[0].pdf_path) as result:
                self.assertIn("123.456.789-09", result[0].get_text())

    def test_generates_pdf_ground_truth_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "test", "dpi": 72, "page_size_px": [300, 300],
                "fields": [{"field_id": "name", "label": "Nome", "data_type": "person_name", "bbox": [20, 20, 280, 60]}],
            }), encoding="utf-8")
            output = root / "output"
            progress = []

            generated = generate_dataset(
                template, mapping, {"name"}, output, 2, seed=7, style_preset_uid="style_test",
                progress_callback=lambda index, total, document: progress.append((index, total, document.pdf_path.name)),
            )

            self.assertEqual(len(generated), 2)
            self.assertEqual(progress, [
                (1, 2, "synthetic_0001__style-style_test.pdf"),
                (2, 2, "synthetic_0002__style-style_test.pdf"),
            ])
            self.assertTrue((output / "dataset.json").is_file())
            manifest = json.loads((output / "dataset.json").read_text())
            self.assertEqual(manifest["style_preset_uid"], "style_test")
            for item in generated:
                self.assertTrue(item.pdf_path.is_file())
                self.assertTrue(item.json_path.is_file())
                self.assertIn("__style-style_test", item.pdf_path.stem)
                self.assertTrue(item.values["name"])
                truth = json.loads(item.json_path.read_text())
                self.assertEqual(truth["style_preset_uid"], "style_test")
                self.assertFalse(truth["visual_filters"]["enabled"])
                with fitz.open(item.pdf_path) as result:
                    self.assertEqual(len(result), 1)

    def test_field_marked_as_printed_overrides_handwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "test", "dpi": 72, "page_size_px": [300, 300],
                "fields": [{
                    "field_id": "typed",
                    "label": "Typed",
                    "data_type": "enum",
                    "bbox": [20, 20, 280, 60],
                    "render_as_printed": True,
                    "validation": {"allowed_values": ["DIGITADO"]},
                }],
            }), encoding="utf-8")

            generated = generate_dataset(
                template, mapping, {"typed"}, root / "output", 1,
                seed=7, handwriting=True, text_handwriting=True,
            )

            with fitz.open(generated[0].pdf_path) as result:
                self.assertIn("DIGITADO", result[0].get_text())

    def test_rejects_empty_field_selection(self):
        with self.assertRaisesRegex(ValueError, "campo"):
            generate_dataset(Path("template.pdf"), Path("mapping.json"), set(), Path("out"), 1)

    def test_generates_checkbox_and_digit_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "compound", "dpi": 72, "page_size_px": [300, 300],
                "fields": [
                    {"field_id": "choice", "label": "Choice", "data_type": "checkbox", "bbox": [10, 10, 100, 40],
                     "selection_mode": "single", "elements": [
                         {"element_id": "yes", "value": "Y", "bbox": [10, 10, 30, 30]},
                         {"element_id": "no", "value": "N", "bbox": [40, 10, 60, 30]}]},
                    {"field_id": "digits", "label": "Digits", "data_type": "digit_sequence", "bbox": [10, 60, 100, 90],
                     "elements": [
                         {"element_id": "d1", "index": 0, "bbox": [10, 60, 30, 85]},
                         {"element_id": "d2", "index": 1, "bbox": [35, 60, 55, 85]}]},
                    {"field_id": "letters", "label": "Letters", "data_type": "text_sequence", "bbox": [10, 100, 100, 130],
                     "elements": [
                         {"element_id": "l1", "index": 0, "bbox": [10, 100, 30, 125]},
                         {"element_id": "l2", "index": 1, "bbox": [35, 100, 55, 125]}]},
                ],
            }), encoding="utf-8")

            generated = generate_dataset(template, mapping, {"choice", "digits", "letters"}, root / "out", 1, seed=2)
            truth = json.loads(generated[0].json_path.read_text())
            manifest = json.loads((root / "out" / "dataset.json").read_text())

        by_id = {field["field_id"]: field for field in truth["fields"]}
        self.assertEqual(sum(element["checked"] for element in by_id["choice"]["elements"]), 1)
        self.assertRegex(by_id["digits"]["value"], r"^\d{2}$")
        self.assertRegex(by_id["letters"]["value"], r"^[A-Z]{2}$")
        self.assertTrue(manifest["handwriting"]["enabled"])

    def test_can_apply_realistic_filters_to_generated_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "filtered", "dpi": 72, "page_size_px": [300, 300],
                "fields": [{"field_id": "name", "label": "Nome", "data_type": "person_name", "bbox": [20, 20, 280, 60]}],
            }), encoding="utf-8")

            generated = generate_dataset(
                template, mapping, {"name"}, root / "out", 1, seed=7,
                realistic_filters=True, filter_preset="light",
            )
            truth = json.loads(generated[0].json_path.read_text())
            manifest = json.loads((root / "out" / "dataset.json").read_text())

            self.assertTrue(truth["visual_filters"]["enabled"])
            self.assertEqual(truth["visual_filters"]["preset"], "light")
            self.assertEqual(manifest["visual_filters"], {"enabled": True, "preset": "light"})
            with fitz.open(generated[0].pdf_path) as result:
                self.assertEqual(len(result), 1)

    def test_uses_dated_document_dataset_folder_when_output_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "documents" / "dnv" / "template.pdf"
            template.parent.mkdir(parents=True)
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            mapping = root / "documents" / "dnv" / "mappings" / "mapping.json"
            mapping.parent.mkdir(parents=True)
            mapping.write_text(json.dumps({
                "template_id": "test", "dpi": 72, "page_size_px": [300, 300],
                "fields": [{"field_id": "name", "label": "Nome", "data_type": "person_name", "bbox": [20, 20, 280, 60]}],
            }), encoding="utf-8")
            output = root / "documents" / "dnv" / "dataset" / "20260915_0807-dnv-1"

            with patch("src.synthetic_dataset.build_dataset_run_dir", return_value=output) as build:
                generated = generate_dataset(template, mapping, {"name"}, None, 1, seed=7)

        build.assert_called_once_with("dnv", 1)
        self.assertEqual(generated[0].pdf_path.parent, output)


if __name__ == "__main__":
    unittest.main()
