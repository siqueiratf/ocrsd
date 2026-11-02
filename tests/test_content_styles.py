"""Fast, offline tests for topic 4 (content-aware writing styles):
- the content classifier (src/content_classes.py)
- Writing/WritingOverride validation and back-compat (src/web/models.py)
- per-content-class font selection in the renderer (src/synthetic_dataset.py),
  font-only so no neural weights are needed
- the preset editor's "Estilo por tipo de conteúdo" section renders
"""
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.content_classes import classify_content
from src.web.models import Writing, WritingOverride
from src.synthetic_dataset import generate_dataset


class ClassifyContentTests(unittest.TestCase):
    def test_checkbox_data_type_wins(self):
        self.assertEqual(classify_content("checkbox", "qualquer.coisa", "X"), "checkbox")

    def test_cpf_by_key(self):
        self.assertEqual(classify_content("text", "pessoa.cpf", "123.456.789-00"), "cpf_cnpj")

    def test_cpf_by_value_shape_without_key(self):
        self.assertEqual(classify_content("text", None, "123.456.789-00"), "cpf_cnpj")

    def test_date_by_data_type(self):
        self.assertEqual(classify_content("date", "pessoa.nascimento", "01/02/2020"), "date")

    def test_name_by_key(self):
        self.assertEqual(classify_content("text", "pessoa.nome", "Maria da Silva"), "name")

    def test_digit_sequence_by_data_type(self):
        self.assertEqual(classify_content("digit_sequence", None, "12345"), "digits")

    def test_place_by_key(self):
        self.assertEqual(classify_content("text", "endereco.municipio", "São Paulo"), "place")

    def test_long_text_by_length(self):
        value = "Este é um texto longo com várias palavras descrevendo algo em detalhe para o campo"
        self.assertEqual(classify_content("text", None, value), "long_text")

    def test_short_text_default(self):
        self.assertEqual(classify_content("text", None, "abc"), "short_text")

    def test_unknown_data_type_defaults_gracefully(self):
        # No exception for an unexpected/empty data_type -- always returns one
        # of the known classes.
        from src.content_classes import CONTENT_CLASSES
        self.assertIn(classify_content("", "", ""), CONTENT_CLASSES)


class WritingModelBackCompatTests(unittest.TestCase):
    def test_default_writing_has_no_content_overrides(self):
        writing = Writing()
        self.assertIsNone(writing.content)

    def test_old_json_without_content_key_parses_unchanged(self):
        # Simulates a job/preset/replay saved before this feature existed.
        old_json = {
            "model": "font", "font_family": "caveat", "handwriting": True, "text_handwriting": True,
            "intensity": 0.65, "writer": None, "bias": 0.75, "color": "#193787", "width": 1.5,
            "pressure": 0.25, "dropout": 0.0, "text_policy": "normalize",
        }
        writing = Writing.model_validate(old_json)
        self.assertIsNone(writing.content)
        # Round-tripping through model_dump must reproduce the same values
        # (plus content=None), so an old replay/job dict stays compatible.
        dumped = writing.model_dump()
        for key, value in old_json.items():
            self.assertEqual(dumped[key], value)
        self.assertIsNone(dumped["content"])

    def test_resolved_for_without_override_returns_base_writing_unchanged(self):
        writing = Writing(font_family="caveat", bias=0.5)
        self.assertIs(writing.resolved_for("digits"), writing)
        self.assertIs(writing.resolved_for("name"), writing)

    def test_resolved_for_with_override_merges_only_overridden_fields(self):
        writing = Writing(font_family="caveat", bias=0.5, color="#123456", content={
            "digits": {"font_family": "courier_prime"},
        })
        resolved = writing.resolved_for("digits")
        self.assertEqual(resolved.font_family, "courier_prime")
        self.assertEqual(resolved.bias, 0.5)  # inherited, not overridden
        self.assertEqual(resolved.color, "#123456")  # inherited

    def test_checkbox_is_not_an_overridable_content_class(self):
        with self.assertRaises(Exception):
            Writing(content={"checkbox": {"font_family": "caveat"}})

    def test_sjvasquez_writer_limit_enforced_in_override(self):
        with self.assertRaises(Exception):
            Writing(content={"digits": {"model": "sjvasquez", "writer": 50}})

    def test_sjvasquez_writer_limit_in_override_uses_base_model_when_override_has_no_model(self):
        # The override only sets writer=50; the effective model comes from
        # the base Writing, which here IS sjvasquez, so this must also fail.
        with self.assertRaises(Exception):
            Writing(model="sjvasquez", content={"digits": {"writer": 50}})

    def test_writing_override_rejects_unknown_fields(self):
        with self.assertRaises(Exception):
            WritingOverride(not_a_field=1)


def _spans(pdf_path):
    with fitz.open(pdf_path) as document:
        page = document[0]
        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans.extend(line["spans"])
        return spans


def _make_template(root: Path) -> Path:
    template = root / "template.pdf"
    document = fitz.open()
    document.new_page(width=1250, height=1250)
    document.save(template)
    document.close()
    return template


def _make_mapping(root: Path) -> Path:
    mapping = root / "mapping.json"
    mapping.write_text(json.dumps({
        "template_id": "content_style", "dpi": 300, "page_size_px": [1250, 1250],
        "fields": [
            {"field_id": "codigo", "label": "Código", "data_type": "digit_sequence",
             "bbox": [40, 40, 300, 100],
             "elements": [{"element_id": f"c{i}", "index": i, "bbox": [40 + i * 30, 40, 70 + i * 30, 100]}
                          for i in range(5)]},
            {"field_id": "nome", "label": "Nome", "data_type": "text", "bbox": [40, 150, 600, 210]},
        ],
    }), encoding="utf-8")
    return mapping


class RendererContentClassSelectionTests(unittest.TestCase):
    """Font-only (no neural weights) tests that the per-content-class font
    override actually changes what gets rendered for that field, while an
    unconfigured document (content_pens/content_fonts=None) renders exactly
    as before."""

    def test_without_overrides_behaviour_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, mapping = _make_template(root), _make_mapping(root)
            generated = generate_dataset(template, mapping, {"codigo", "nome"}, root / "out", 1, seed=1,
                                         handwriting=False, text_handwriting=False, font_family="arimo")
            spans = _spans(generated[0].pdf_path)
            # Typed font family ("arimo") for the whole document -> everything
            # is printed text using the typed font alias, nothing else.
            self.assertTrue(spans)
            for span in spans:
                self.assertEqual(span["font"], "Arimo-Regular")

    def test_content_font_override_changes_only_that_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, mapping = _make_template(root), _make_mapping(root)
            generated = generate_dataset(
                template, mapping, {"codigo", "nome"}, root / "out", 1, seed=1,
                handwriting=False, text_handwriting=False, font_family="arimo",
                content_pens={"digits": None}, content_fonts={"digits": "tinos"},
            )
            spans = _spans(generated[0].pdf_path)
            self.assertTrue(spans)
            fonts_used = {span["font"] for span in spans}
            self.assertIn("Tinos-Regular", fonts_used)  # the "codigo" field, overridden
            self.assertIn("Arimo-Regular", fonts_used)  # the "nome" field, base font family

    def test_content_handwriting_false_falls_back_to_plain_print(self):
        # A content class with a font-family override but handwriting
        # explicitly turned off (content_handwriting[cls] = False) must not
        # render as handwriting -- it should fall through to the plain
        # printed fallback (fontname "helv"), not the cursive/numeric
        # handwriting image path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, mapping = _make_template(root), _make_mapping(root)
            generated = generate_dataset(
                template, mapping, {"codigo", "nome"}, root / "out", 1, seed=1,
                handwriting=False, text_handwriting=False, font_family="caveat",
                content_pens={"digits": None}, content_fonts={"digits": "caveat"},
                content_handwriting={"digits": False},
            )
            spans = _spans(generated[0].pdf_path)
            fonts_used = {span["font"] for span in spans}
            self.assertIn("Helvetica", fonts_used)  # "codigo" fell back to plain print

    def test_deterministic_with_content_overrides(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            results = []
            for directory in (d1, d2):
                root = Path(directory)
                template, mapping = _make_template(root), _make_mapping(root)
                generated = generate_dataset(
                    template, mapping, {"codigo", "nome"}, root / "out", 1, seed=7,
                    handwriting=False, text_handwriting=False, font_family="arimo",
                    content_pens={"digits": None}, content_fonts={"digits": "tinos"},
                )
                results.append(json.loads(generated[0].json_path.read_text(encoding="utf-8"))["fields"])
            values_a = [(f["field_id"], f["value"]) for f in results[0]]
            values_b = [(f["field_id"], f["value"]) for f in results[1]]
            self.assertEqual(values_a, values_b)


class PresetEditorTemplateTests(unittest.TestCase):
    def test_template_mentions_content_style_section(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "web" / "templates" / "preset_editor.html").read_text(encoding="utf-8")
        self.assertIn("Estilo por tipo de conteúdo", source)
        self.assertIn("content_classes", source)

    def test_content_overrides_state_parses_font_rows(self):
        from starlette.datastructures import FormData
        from src.web.ui import content_overrides_state

        form = FormData([
            ("content_digits_model", "font"), ("content_digits_font_family", "courier_prime"),
            ("content_digits_intensity", "0.65"),
            ("content_name_font_family", "caveat"),  # model left blank -> "Igual ao principal", ignored
        ])
        overrides = content_overrides_state(form)
        self.assertEqual(set(overrides), {"digits"})
        self.assertEqual(overrides["digits"], {"model": "font", "font_family": "courier_prime", "intensity": 0.65})

    def test_content_overrides_state_parses_neural_rows(self):
        from starlette.datastructures import FormData
        from src.web.ui import content_overrides_state

        form = FormData([
            ("content_name_model", "sjvasquez"), ("content_name_writer", "4"), ("content_name_bias", "2.5"),
        ])
        overrides = content_overrides_state(form)
        self.assertEqual(overrides, {"name": {"model": "sjvasquez", "writer": 4, "bias": 2.5}})

    def test_content_overrides_state_empty_when_nothing_enabled(self):
        from starlette.datastructures import FormData
        from src.web.ui import content_overrides_state

        self.assertEqual(content_overrides_state(FormData([])), {})


class RenderWritingSamplesContentClassTests(unittest.TestCase):
    """Font-only (no neural weights) coverage for the preview's per-class
    resolution: render_writing_samples() must use the class's own override
    (when one exists for that sample's content_class) instead of the base
    style, and leave unconfigured samples on the base style."""

    def test_sample_without_matching_override_uses_base_style(self):
        from src.web.services import render_writing_samples

        writing = Writing(font_family="caveat", content={"digits": {"model": "font", "font_family": "courier_prime"}})
        results = render_writing_samples(writing, ["Ana Silva"], content_classes=["name"])
        self.assertFalse(results[0]["overridden"])

    def test_sample_with_matching_override_is_flagged_overridden(self):
        from src.web.services import render_writing_samples

        writing = Writing(font_family="caveat", content={"digits": {"model": "font", "font_family": "courier_prime"}})
        results = render_writing_samples(writing, ["048125"], content_classes=["digits"])
        self.assertTrue(results[0]["overridden"])
        self.assertEqual(results[0]["content_class"], "digits")


if __name__ == "__main__":
    unittest.main()
