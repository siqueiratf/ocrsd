"""Fast, offline tests for the typed-text (printed-font) and font-handwriting
placement rules in src/synthetic_dataset.py (left padding, vertical centring,
size-from-height, shrink-to-fit floor). No neural models involved.
"""
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.fonts import font_path
from src.synthetic_dataset import (
    TYPED_GENERIC_MAX_SIZE, TYPED_HARD_FLOOR, TYPED_MAX_FONT_SIZE, TYPED_MIN_FONT_SIZE, TYPED_SHRINK_FLOOR,
    _typed_digit_height_ratio, generate_dataset, render_handwritten_text,
)


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


class TypedDigitHeightRatioTests(unittest.TestCase):
    def test_ratio_is_plausible_for_each_typed_font(self):
        for family in ("courier_prime", "arimo", "tinos"):
            ratio = _typed_digit_height_ratio(str(font_path(family)))
            self.assertTrue(0.3 <= ratio <= 1.0, f"{family}: {ratio}")

    def test_unknown_font_file_falls_back_safely(self):
        self.assertAlmostEqual(_typed_digit_height_ratio("/no/such/font.ttf"), 0.62)


class TypedFontSizeScalesWithHeightTests(unittest.TestCase):
    def _font_size_for(self, root: Path, height_px: int, font_family: str = "arimo") -> float:
        template = _make_template(root)
        mapping = root / "mapping.json"
        # A composite digit box (single-character element) isolates the size formula
        # from the shrink-to-fit-by-width loop that a long string could trigger.
        mapping.write_text(json.dumps({
            "template_id": "size", "dpi": 300, "page_size_px": [1250, 1250],
            "fields": [{
                "field_id": "digit", "label": "D", "data_type": "digit_sequence",
                "bbox": [40, 40, 40 + height_px, 40 + height_px],
                "elements": [{"element_id": "d0", "index": 0, "bbox": [40, 40, 40 + height_px, 40 + height_px]}],
            }],
        }), encoding="utf-8")
        generated = generate_dataset(template, mapping, {"digit"}, root / "out", 1, seed=1, font_family=font_family)
        spans = _spans(generated[0].pdf_path)
        self.assertEqual(len(spans), 1)
        return spans[0]["size"]

    def test_tall_field_is_not_capped_at_eleven_points(self):
        with tempfile.TemporaryDirectory() as directory:
            size = self._font_size_for(Path(directory), height_px=120)
            # Old formula: max(5, min(11, height*point_scale*0.62)) == 11 for this height.
            self.assertGreater(size, 11.0)

    def test_font_size_increases_with_field_height(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            small = self._font_size_for(Path(d1), height_px=40)
            large = self._font_size_for(Path(d2), height_px=140)
        self.assertGreater(large, small)

    def test_font_size_respects_sensible_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            size = self._font_size_for(Path(directory), height_px=2000)
        self.assertLessEqual(size, TYPED_MAX_FONT_SIZE + 1e-6)

    def test_font_size_respects_minimum_or_blanks_rather_than_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            height_px = 2
            mapping.write_text(json.dumps({
                "template_id": "size", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "digit", "label": "D", "data_type": "digit_sequence",
                    "bbox": [40, 40, 40 + height_px, 40 + height_px],
                    "elements": [{"element_id": "d0", "index": 0, "bbox": [40, 40, 40 + height_px, 40 + height_px]}],
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"digit"}, root / "out", 1, seed=1, font_family="arimo")
            spans = _spans(generated[0].pdf_path)
            # A 2px-tall cell cannot fit anything legibly: the placement engine must
            # never draw clipped ink, so either nothing is drawn (blank-with-reason) or,
            # if it is, the size used is never below the absolute hard floor.
            if spans:
                self.assertGreaterEqual(spans[0]["size"], TYPED_HARD_FLOOR - 1e-6)
            else:
                ground_truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
                entry = next(item for item in ground_truth["fields"] if item["field_id"] == "digit")
                self.assertEqual(entry["value"], "")


class TypedTextPlacementTests(unittest.TestCase):
    def test_multi_char_text_has_left_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            field_x0, field_y0, field_x1 = 40, 40, 900
            mapping.write_text(json.dumps({
                "template_id": "pad", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "name", "label": "Nome", "data_type": "person_name",
                    "render_as_printed": True,
                    "bbox": [field_x0, field_y0, field_x1, field_y0 + 60],
                    "validation": {"allowed_values": ["ANA SILVA"]},
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"name"}, root / "out", 1, seed=1, font_family="arimo")
            spans = _spans(generated[0].pdf_path)
            self.assertEqual(len(spans), 1)
            point_scale = 72.0 / 300.0
            field_x0_pt = field_x0 * point_scale
            field_height_pt = 60 * point_scale
            offset = spans[0]["bbox"][0] - field_x0_pt
            # Old behaviour started essentially flush with the field's left edge
            # (left_margin_w ~0.018-0.02 in the writing audit); new rule pads by
            # TYPED_LEFT_PAD_HEIGHT_FRAC (0.18) of the field height.
            self.assertGreater(offset, 0.1 * field_height_pt)
            self.assertLess(offset, 0.3 * field_height_pt)

    def test_single_character_element_stays_centered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "template_id": "centered", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "digit", "label": "D", "data_type": "digit_sequence",
                    "bbox": [40, 40, 140, 140],
                    "elements": [{"element_id": "d0", "index": 0, "bbox": [40, 40, 140, 140]}],
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"digit"}, root / "out", 1, seed=3, font_family="courier_prime")
            spans = _spans(generated[0].pdf_path)
            self.assertEqual(len(spans), 1)
            point_scale = 72.0 / 300.0
            rect = fitz.Rect(40 * point_scale, 40 * point_scale, 140 * point_scale, 140 * point_scale)
            bbox = spans[0]["bbox"]
            glyph_center_x = (bbox[0] + bbox[2]) / 2
            rect_center_x = (rect.x0 + rect.x1) / 2
            self.assertLess(abs(glyph_center_x - rect_center_x), rect.width * 0.15)

    def test_long_value_that_cannot_fit_is_left_blank_with_a_reason_not_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            long_value = "X" * 60
            mapping.write_text(json.dumps({
                "template_id": "overflow", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "name", "label": "Nome", "data_type": "person_name",
                    "render_as_printed": True,
                    "bbox": [40, 40, 160, 100],
                    "validation": {"allowed_values": [long_value]},
                }],
            }), encoding="utf-8")
            # Must not raise, and must not draw clipped text: the visible PDF text
            # must always equal the ground-truth value, so an unfittable value is
            # recorded as blank-with-reason instead of being drawn over the edge.
            generated = generate_dataset(template, mapping, {"name"}, root / "out", 1, seed=5, font_family="arimo")
            ground_truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
            entry = next(item for item in ground_truth["fields"] if item["field_id"] == "name")
            self.assertTrue(entry.get("blank"))
            self.assertEqual(entry["value"], "")
            self.assertEqual(entry.get("requested_value"), long_value)
            spans = _spans(generated[0].pdf_path)
            self.assertEqual(len(spans), 0)

    def test_value_that_fits_at_the_hard_floor_is_drawn_not_blanked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            # Long enough to need the hard floor, short enough to still fit there.
            value = "AVENIDA GOIAS SETOR CENTRAL"
            mapping.write_text(json.dumps({
                "template_id": "tight", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "addr", "label": "End.", "data_type": "address",
                    "render_as_printed": True,
                    "bbox": [40, 40, 920, 92],
                    "validation": {"allowed_values": [value]},
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"addr"}, root / "out", 1, seed=5, font_family="courier_prime")
            ground_truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
            entry = next(item for item in ground_truth["fields"] if item["field_id"] == "addr")
            self.assertEqual(entry["value"], value)
            self.assertFalse(entry.get("blank"))
            spans = _spans(generated[0].pdf_path)
            self.assertEqual(len(spans), 1)
            self.assertGreaterEqual(spans[0]["size"], TYPED_HARD_FLOOR - 1e-6)


class HandwritingLeftAlignmentTests(unittest.TestCase):
    def test_multi_character_text_starts_near_left_edge(self):
        import random

        image = render_handwritten_text("ANA", 400, 100, random.Random(7), medium="pencil")
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        self.assertIsNotNone(bbox)
        self.assertLess(bbox[0], 400 * 0.25)

    def test_single_character_stays_roughly_centered(self):
        import random

        image = render_handwritten_text("A", 200, 100, random.Random(7), medium="pencil")
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        self.assertIsNotNone(bbox)
        center = (bbox[0] + bbox[2]) / 2
        self.assertLess(abs(center - 100), 100 * 0.3)


class TypedGenericSizeCapTests(unittest.TestCase):
    """Reproduces the verifier's CNPJ quadro-de-socios finding: a tall "fill-in-the-
    line" multi-character field must not balloon to the single-character-cell max size,
    and same-geometry fields must land in a realistic, consistent size range regardless
    of how long the value happens to be."""

    def _size_for(self, root: Path, value: str, bbox, field_id="field") -> float | None:
        template = _make_template(root)
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps({
            "template_id": "cap", "dpi": 300, "page_size_px": [1250, 1250],
            "fields": [{
                "field_id": field_id, "label": "L", "data_type": "text",
                "render_as_printed": True, "bbox": list(bbox),
                "validation": {"allowed_values": [value]},
            }],
        }), encoding="utf-8")
        generated = generate_dataset(template, mapping, {field_id}, root / "out", 1, seed=1, font_family="arimo")
        spans = _spans(generated[0].pdf_path)
        return spans[0]["size"] if spans else None

    def test_tall_multi_character_field_is_capped_to_a_realistic_size(self):
        with tempfile.TemporaryDirectory() as directory:
            # A tall table-row cell (height >> what one typed line needs), like the
            # CNPJ quadro-de-socios "Qualificacao" column.
            size = self._size_for(Path(directory), "Socio", bbox=[100, 100, 400, 220])
        self.assertIsNotNone(size)
        self.assertLessEqual(size, TYPED_GENERIC_MAX_SIZE + 1e-6)

    def test_short_and_long_values_in_same_tall_box_land_in_a_consistent_range(self):
        bbox = [100, 100, 500, 220]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            short_size = self._size_for(Path(d1), "Socio", bbox)
            long_size = self._size_for(Path(d2), "Socio-Administrador", bbox)
        self.assertIsNotNone(short_size)
        self.assertIsNotNone(long_size)
        # Before the fix: the short value kept the full height-derived (huge) size while
        # the long one shrank to fit, so the two could differ by >3x. After the cap,
        # both start from the same capped target size and only the long one may need to
        # shrink a little further -- they should stay within a realistic typed range of
        # each other.
        self.assertLessEqual(short_size, TYPED_GENERIC_MAX_SIZE + 1e-6)
        self.assertLess(short_size / max(long_size, 1e-6), 2.0)

    def test_single_character_cell_is_unaffected_by_the_generic_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            height_px = 120
            mapping.write_text(json.dumps({
                "template_id": "digitcap", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "digit", "label": "D", "data_type": "digit_sequence",
                    "bbox": [40, 40, 40 + height_px, 40 + height_px],
                    "elements": [{"element_id": "d0", "index": 0, "bbox": [40, 40, 40 + height_px, 40 + height_px]}],
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"digit"}, root / "out", 1, seed=1, font_family="arimo")
            spans = _spans(generated[0].pdf_path)
        self.assertEqual(len(spans), 1)
        # A composite digit-box cell this tall legitimately needs a size above the
        # generic cap -- it must not be clamped to TYPED_GENERIC_MAX_SIZE.
        self.assertGreater(spans[0]["size"], TYPED_GENERIC_MAX_SIZE)

    def test_descenders_do_not_cross_the_field_bottom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _make_template(root)
            mapping = root / "mapping.json"
            # A short, wide, tall box with a descender-heavy value: real risk case for
            # an underline-style "fill in the line" field.
            field_y0, field_y1 = 100, 220
            mapping.write_text(json.dumps({
                "template_id": "descenders", "dpi": 300, "page_size_px": [1250, 1250],
                "fields": [{
                    "field_id": "addr", "label": "L", "data_type": "address",
                    "render_as_printed": True, "bbox": [100, field_y0, 900, field_y1],
                    "validation": {"allowed_values": ["Rua Principal, 123 - Jardim"]},
                }],
            }), encoding="utf-8")
            generated = generate_dataset(template, mapping, {"addr"}, root / "out", 1, seed=1, font_family="arimo")
            spans = _spans(generated[0].pdf_path)
        self.assertEqual(len(spans), 1)
        point_scale = 72.0 / 300.0
        field_bottom_pt = field_y1 * point_scale
        self.assertLessEqual(spans[0]["bbox"][3], field_bottom_pt + 1e-3)


if __name__ == "__main__":
    unittest.main()
