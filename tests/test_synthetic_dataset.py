import json
import random
from datetime import date
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from src.synthetic_dataset import (
    _apply_ballpoint_texture, apply_shared_rules, element_values, generate_dataset, is_numeric_text,
    random_handwriting_style, random_value, render_checkbox_mark,
    render_handwritten_numeric, render_handwritten_text, render_neural_text, typed_value,
)
from src.field_rules import DNV_PATTERN, cpf, dnv_structure_errors, format_date, rule_errors
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

    def test_generates_editor_field_types_and_rules(self):
        rng = random.Random(11)
        self.assertRegex(random_value({"data_type": "date", "validation": {"format": "AAAA-MM-DD"}}, rng), r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(random_value({"data_type": "date", "validation": {"format": "DD de mês de AAAA"}}, rng), r"^\d{2} de [a-zç]+ de \d{4}$")
        self.assertRegex(random_value({"data_type": "text", "validation": {"mask": "AA-####"}}, rng), r"^[A-Z]{2}-\d{4}$")
        self.assertRegex(random_value({"data_type": "document", "validation": {"document_kind": "rg"}}, rng), r"^\d{2}\.\d{3}\.\d{3}-\d$")
        self.assertEqual(len(random_value({"data_type": "document", "validation": {"document_kind": "cns"}}, rng)), 15)
        decimal = float(random_value({"data_type": "decimal", "validation": {"min_value": 10, "max_value": 12, "decimals": 1}}, rng).replace(",", "."))
        self.assertTrue(10 <= decimal <= 12)
        self.assertRegex(random_value({"data_type": "decimal"}, rng), r"^\d,\d{3}$")
        for data_type in ("municipality", "address", "occupation", "signature"):
            self.assertTrue(random_value({"data_type": data_type}, rng))
        self.assertIn(random_value({"data_type": "municipality", "validation": {"allowed_values": ["Lages"]}}, rng), {"Lages"})

    def test_generates_valid_cpf_check_digits(self):
        for seed in range(20):
            digits = [int(character) for character in cpf(random.Random(seed)) if character.isdigit()]
            for length in (9, 10):
                total = sum(value * weight for value, weight in zip(digits, range(length + 1, 1, -1)))
                self.assertEqual(digits[length], 0 if total % 11 < 2 else 11 - total % 11)

    def test_validates_declarative_rules(self):
        self.assertEqual(rule_errors("text", {"mask": "###", "allowed_values": ["A"], "blank_probability": 0.5}), [])
        self.assertTrue(rule_errors("integer", {"min_value": 5, "max_value": 1}))
        self.assertTrue(rule_errors("text", {"mask": "sem tokens"}))
        self.assertTrue(rule_errors("date", {"format": "%Y"}))
        self.assertTrue(rule_errors("text", {"pattern": "("}))
        self.assertTrue(rule_errors("text", {"blank_probability": 1}))
        self.assertTrue(rule_errors("document", {"document_kind": "passaporte"}))
        self.assertEqual(rule_errors("digit_sequence", {"format": "DDMMYYYY"}), [])

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

    def test_field_rules_take_precedence_over_shared_rules(self):
        resolved = apply_shared_rules({"shared_rules": {"RULE_UF": {"description": "UF", "allowed_values": ["SP", "RJ"], "max_edit_distance": 1}},
                                       "fields": [{"field_id": "uf", "rule_refs": ["RULE_UF"], "validation": {"allowed_values": ["SC"]}}]})
        self.assertEqual(resolved["fields"][0]["validation"], {"allowed_values": ["SC"], "max_edit_distance": 1})

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

    def test_location_digit_sequences_keep_only_digits(self):
        field = {"field_id": "12_codigo", "description": "Municipio de ocorrencia - Codigo (IBGE)", "data_type": "digit_sequence",
                 "elements": [{"index": index} for index in range(7)]}
        for seed in range(40):
            context: dict = {}
            value = random_value(field, random.Random(seed), context)
            self.assertRegex(value, r"^[0-9]{7}$")
            self.assertEqual(value, context["locations"]["occurrence"]["ibge"])

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


class TypedFieldTests(unittest.TestCase):
    def test_date_formats_describe_one_valid_date(self):
        patterns = {"DD/MM/AAAA": r"^\d{2}/\d{2}/\d{4}$", "DD-MM-AAAA": r"^\d{2}-\d{2}-\d{4}$",
                    "DD de mês de AAAA": r"^\d{2} de (janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro) de \d{4}$"}
        for fmt, pattern in patterns.items():
            for seed in range(40):
                value, semantic = typed_value({"field_id": "d", "data_type": "date", "validation": {"format": fmt}}, random.Random(seed), {})
                self.assertRegex(value, pattern)
                self.assertEqual(value, format_date(date.fromisoformat(semantic["date"]), fmt))

    def test_dates_written_fully_in_words(self):
        self.assertEqual(format_date(date(2021, 2, 3), "EXTENSO_PALAVRAS"), "três de fevereiro de dois mil e vinte e um")
        self.assertEqual(format_date(date(1999, 12, 31), "EXTENSO_PALAVRAS"), "trinta e um de dezembro de mil novecentos e noventa e nove")
        self.assertEqual(format_date(date(1900, 1, 1), "EXTENSO_PALAVRAS"), "primeiro de janeiro de mil e novecentos")
        self.assertEqual(format_date(date(2024, 2, 29), "EXTENSO_PALAVRAS"), "vinte e nove de fevereiro de dois mil e vinte e quatro")
        from src.semantic.present import present_text
        from src.semantic.resolve import Binding
        field = {"field_id": "data_extenso", "data_type": "text", "validation": {"pattern": "^[A-Za-zÀ-ÖØ-öø-ÿ' -]+$"}}
        binding = Binding("birth.date", format="DD de mês de AAAA")
        self.assertEqual(present_text(field, binding, date(2021, 2, 3)), "três de fevereiro de dois mil e vinte e um")
        self.assertEqual(present_text({**field, "validation": {}}, binding, date(2021, 2, 3)), "03 de fevereiro de 2021")

    def test_leap_days_are_generated_and_formatted(self):
        self.assertEqual(format_date(date(2024, 2, 29), "DD-MM-AAAA"), "29-02-2024")
        self.assertEqual(format_date(date(2024, 2, 29), "DD de mês de AAAA"), "29 de fevereiro de 2024")
        field = {"field_id": "d", "data_type": "date", "validation": {"format": "DD/MM/AAAA"}}
        leap = next(semantic["date"] for seed in range(20000)
                    for value, semantic in [typed_value(field, random.Random(seed), {})] if semantic["date"].endswith("-02-29"))
        self.assertEqual(date.fromisoformat(leap).day, 29)

    def test_separated_date_parts_come_from_one_date(self):
        field = {"field_id": "nascimento", "data_type": "date", "validation": {"format": "SEPARADO"}, "elements": [
            {"element_id": "ano", "part": "year", "bbox": [60, 0, 100, 10]},
            {"element_id": "dia", "part": "day", "bbox": [0, 0, 20, 10]},
            {"element_id": "mes", "part": "month", "bbox": [30, 0, 50, 10]}]}
        for seed in range(30):
            value, parts, semantic = element_values(field, random.Random(seed), {})
            generated = date.fromisoformat(semantic["date"])
            self.assertEqual(value, generated.strftime("%d/%m/%Y"))
            self.assertEqual({element["part"]: text for element, text in parts},
                             {"day": f"{generated.day:02d}", "month": f"{generated.month:02d}", "year": str(generated.year)})

    def test_sex_representations_share_one_value(self):
        for seed in range(20):
            context: dict = {}
            full, first = typed_value({"field_id": "sexo", "data_type": "sex", "validation": {"format": "completo"}}, random.Random(seed), context)
            short, second = typed_value({"field_id": "sexo_rn", "data_type": "sex", "validation": {"format": "abreviado"}}, random.Random(seed + 1), context)
            self.assertIn(full, {"Masculino", "Feminino"})
            self.assertEqual(short, first["sex"])
            self.assertEqual(first, second)
            self.assertEqual(full[0], short)
        self.assertTrue(rule_errors("sex", {"format": "outro"}))

    def test_synthetic_dnv_identifier_uses_documented_structure(self):
        grouped, semantic = typed_value({"field_id": "dnv", "data_type": "dnv_number", "validation": {"format": "agrupado"}}, random.Random(4), {})
        self.assertRegex(grouped, DNV_PATTERN)
        self.assertEqual(dnv_structure_errors(grouped), [])
        self.assertEqual(grouped[:2], "00")
        self.assertFalse(semantic["identifier"]["official"])
        self.assertTrue(semantic["identifier"]["synthetic"])
        self.assertEqual(grouped.replace("-", ""), semantic["identifier"]["prefix"] + semantic["identifier"]["sequence"] + semantic["identifier"]["check_digit"])
        digits, _ = typed_value({"field_id": "dnv", "data_type": "dnv_number", "validation": {"format": "digitos"}}, random.Random(4), {})
        self.assertRegex(digits, r"^00\d{9}$")
        boxes = {"field_id": "dnv", "data_type": "dnv_number", "validation": {"format": "caixas"},
                 "elements": [{"element_id": f"d{index}", "index": index, "bbox": [index * 10, 0, index * 10 + 8, 10]} for index in range(11)]}
        value, parts, _ = element_values(boxes, random.Random(4), {})
        self.assertEqual("".join(text for _, text in parts), value)
        self.assertEqual(len(parts), 11)
        self.assertTrue(dnv_structure_errors("00-1234-5"))
        self.assertTrue(rule_errors("dnv_number", {"prefix": "7"}))


class SyntheticDatasetTests(unittest.TestCase):
    def test_ground_truth_keeps_typed_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page(width=400, height=300)
                document.save(template)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"template_id": "tipos", "dpi": 72, "page_size_px": [400, 300], "fields": [
                {"field_id": "data", "label": "Data", "data_type": "date", "bbox": [10, 10, 200, 40],
                 "validation": {"format": "SEPARADO", "semantic_key": "nascimento"}, "elements": [
                     {"element_id": "dia", "part": "day", "bbox": [10, 10, 40, 40]},
                     {"element_id": "mes", "part": "month", "bbox": [50, 10, 80, 40]},
                     {"element_id": "ano", "part": "year", "bbox": [90, 10, 160, 40]}]},
                {"field_id": "data_extenso", "label": "Data por extenso", "data_type": "date", "bbox": [10, 50, 390, 80],
                 "validation": {"format": "DD de mês de AAAA", "semantic_key": "nascimento"}},
                {"field_id": "sexo", "label": "Sexo", "data_type": "sex", "bbox": [10, 90, 200, 120], "validation": {"format": "completo"}},
                {"field_id": "sexo_sigla", "label": "Sexo sigla", "data_type": "sex", "bbox": [210, 90, 250, 120], "validation": {"format": "abreviado"}},
                {"field_id": "dnv", "label": "DNV", "data_type": "dnv_number", "bbox": [10, 130, 390, 170], "validation": {"format": "agrupado"}},
            ]}), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"data", "data_extenso", "sexo", "sexo_sigla", "dnv"}, root / "out", 3, seed=11)
            for document in generated:
                truth = {field["field_id"]: field for field in json.loads(document.json_path.read_text())["fields"]}
                self.assertEqual(truth["data"]["date"], truth["data_extenso"]["date"])
                when = date.fromisoformat(truth["data"]["date"])
                self.assertEqual([element["value"] for element in truth["data"]["elements"]], [f"{when.day:02d}", f"{when.month:02d}", str(when.year)])
                self.assertEqual(truth["data_extenso"]["value"], format_date(when, "DD de mês de AAAA"))
                self.assertEqual(truth["sexo"]["sex"], truth["sexo_sigla"]["sex"])
                self.assertEqual(truth["sexo_sigla"]["value"], truth["sexo"]["sex"])
                self.assertRegex(truth["dnv"]["value"], DNV_PATTERN)
                self.assertFalse(truth["dnv"]["identifier"]["official"])

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

    def test_blank_probability_leaves_optional_fields_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page(width=300, height=300)
                document.save(template)
            mapping = root / "mapping.json"
            fields = [{"field_id": "optional", "label": "Opcional", "data_type": "text", "validation": {"blank_probability": 0.95}, "bbox": [20, 20, 280, 60]},
                      {"field_id": "always", "label": "Sempre", "data_type": "occupation", "bbox": [20, 80, 280, 120]}]
            mapping.write_text(json.dumps({"template_id": "test", "dpi": 72, "page_size_px": [300, 300], "fields": fields}), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"optional", "always"}, root / "out", 6, seed=3)
            blanks = [item.values["optional"] for item in generated].count("")
            self.assertGreaterEqual(blanks, 4)
            self.assertTrue(all(item.values["always"] for item in generated))
            truth = json.loads(generated[0].json_path.read_text())
            self.assertEqual({entry["field_id"] for entry in truth["fields"]}, {"optional", "always"})

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

    def test_neural_pen_leaves_blank_values_empty(self):
        model = unittest.mock.Mock()
        model.sample.return_value = np.array([[0, 0, 0], [10, 1, 0], [20, 0, 1]], dtype=float)
        with tempfile.TemporaryDirectory() as directory, patch("src.synthetic_dataset.load_model", return_value=model):
            root = Path(directory)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page(width=300, height=300)
                document.save(template)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "blank", "dpi": 72, "page_size_px": [300, 300],
                "fields": [{"field_id": "blank", "label": "Blank", "bbox": [10, 10, 200, 40], "validation": {"allowed_values": [" "]}},
                           {"field_id": "number", "label": "Number", "bbox": [10, 60, 200, 90], "validation": {"allowed_values": ["123"]}}],
            }), encoding="utf-8")

            generated = generate_dataset(template, mapping, {"blank", "number"}, root / "out", 1, seed=3,
                                         neural_pen=NeuralPen(model="aaron1259", text_policy="normalize"))
            truth = json.loads(generated[0].json_path.read_text())

        values = {field["field_id"]: field["value"] for field in truth["fields"]}
        self.assertEqual(values, {"blank": " ", "number": "123"})
        self.assertEqual(model.sample.call_count, 1)

    def test_rejects_empty_field_selection(self):
        with self.assertRaisesRegex(ValueError, "campo"):
            generate_dataset(Path("template.pdf"), Path("mapping.json"), set(), Path("out"), 1)

    def test_single_box_checkbox_can_be_marked_or_left_empty(self):
        from src.web.services import mapping_issues

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.pdf"
            document = fitz.open()
            document.new_page(width=300, height=300)
            document.save(template)
            document.close()
            field = {"field_id": "ignorado", "label": "Ignorado", "data_type": "checkbox", "bbox": [10, 10, 60, 40],
                     "elements": [{"element_id": "ignorado_caixa", "label": "Ignorado", "value": "Ignorado", "bbox": [12, 12, 32, 32]}]}
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"template_id": "single_box", "dpi": 72, "page_size_px": [300, 300], "fields": [field]}))
            self.assertEqual(mapping_issues({"fields": [field]}), [])
            generated = generate_dataset(template, mapping, {"ignorado"}, root / "out", 12, seed=7,
                                         handwriting=False, text_handwriting=False)
            values = [item.values["ignorado"] for item in generated]
            self.assertIn("", values)
            self.assertIn("Ignorado", values)

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
