import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from writing_audit import (  # noqa: E402
    analyze_ink, cer, check_coherence, checkbox_mark_metrics, classify_field_shape,
    cpf_check_digits_ok, normalize_loose, validate_value,
)
from PIL import Image  # noqa: E402


class WritingAuditMetricsTest(unittest.TestCase):
    def test_analyze_ink_no_ink(self):
        crop = np.full((40, 100, 3), 255, dtype=np.uint8)
        metrics = analyze_ink(crop, None, field_h_px=40, field_w_px=100)
        self.assertFalse(metrics.has_ink)

    def test_analyze_ink_placement(self):
        crop = np.full((40, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 20:40, :] = 0  # one dark block, left-ish, roughly vertically centered
        metrics = analyze_ink(crop, None, field_h_px=40, field_w_px=100)
        self.assertTrue(metrics.has_ink)
        self.assertAlmostEqual(metrics.left_margin_h, 20 / 40, places=2)
        self.assertAlmostEqual(metrics.left_margin_w, 20 / 100, places=2)
        self.assertAlmostEqual(metrics.ink_height_ratio, 20 / 40, places=2)
        self.assertFalse(metrics.right_overflow)

    def test_analyze_ink_right_overflow(self):
        crop = np.full((40, 100, 3), 255, dtype=np.uint8)
        crop[10:30, 90:100, :] = 0
        metrics = analyze_ink(crop, None, field_h_px=40, field_w_px=100)
        self.assertTrue(metrics.right_overflow)

    def test_gap_detection_excessive_spacing(self):
        crop = np.full((20, 200, 3), 255, dtype=np.uint8)
        crop[5:15, 0:10, :] = 0      # chunk 1
        crop[5:15, 15:25, :] = 0     # chunk 2, small gap
        crop[5:15, 150:160, :] = 0   # chunk 3, huge gap -> excessive
        metrics = analyze_ink(crop, None, field_h_px=20, field_w_px=200)
        self.assertTrue(metrics.excessive_spacing)
        self.assertGreater(metrics.max_gap_over_ink_height, 1.5)

    def test_gap_detection_normal_spacing(self):
        crop = np.full((20, 60, 3), 255, dtype=np.uint8)
        crop[5:15, 0:10, :] = 0
        crop[5:15, 14:24, :] = 0
        crop[5:15, 28:38, :] = 0
        metrics = analyze_ink(crop, None, field_h_px=20, field_w_px=60)
        self.assertFalse(metrics.excessive_spacing)

    def test_diff_against_blank_template_removes_printed_ink(self):
        blank = np.full((20, 40, 3), 255, dtype=np.uint8)
        blank[0:20, 0:2, :] = 0  # a printed vertical rule (part of the form, not handwriting)
        crop = blank.copy()
        crop[5:15, 20:30, :] = 0  # handwritten ink added on top
        metrics = analyze_ink(crop, blank, field_h_px=20, field_w_px=40)
        # ink mask should only pick the handwritten block, not the pre-existing rule
        self.assertTrue(metrics.has_ink)
        self.assertGreater(metrics.left_margin_h, 0.3)
        self.assertGreater(metrics.left_margin_w, 0.1)

    def test_field_shape_classification(self):
        self.assertEqual(classify_field_shape("checkbox", True, 3, [0, 0, 100, 20]), "checkbox")
        self.assertEqual(classify_field_shape("digit_sequence", False, 8, [0, 0, 200, 30]), "composite_box")
        self.assertEqual(classify_field_shape("text", False, 0, [0, 0, 200, 30]), "single_line")
        self.assertEqual(classify_field_shape("address", False, 0, [0, 0, 400, 150]), "multiline")

    def test_checkbox_mark_metrics_detects_mark(self):
        blank = Image.new("RGB", (60, 60), "white")
        page = blank.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(page)
        draw.line((25, 25, 35, 35), fill=(0, 0, 0), width=2)
        draw.line((35, 25, 25, 35), fill=(0, 0, 0), width=2)
        geo = checkbox_mark_metrics(page, blank, [20, 20, 40, 40])
        self.assertTrue(geo["mark_present"])
        self.assertIsNotNone(geo["mark_angle_deg"])
        self.assertIsNotNone(geo["mark_stroke_width_px"])

    def test_checkbox_mark_metrics_no_mark(self):
        blank = Image.new("RGB", (60, 60), "white")
        geo = checkbox_mark_metrics(blank.copy(), blank, [20, 20, 40, 40])
        self.assertFalse(geo["mark_present"])

    def test_coherence_flags_birth_after_death(self):
        snapshot = {"deceased.birth_date": "2000-01-01", "death.date": "1999-01-01",
                    "death_registration.date": "1999-01-05"}
        issues = check_coherence(snapshot)
        self.assertIn("data_anterior_ao_nascimento_da_pessoa", issues)

    def test_coherence_clean_record(self):
        snapshot = {"deceased.birth_date": "1950-01-01", "deceased.age": 49, "death.date": "1999-06-15",
                    "death_registration.date": "1999-06-20"}
        issues = check_coherence(snapshot)
        self.assertEqual(issues, [])

    def test_coherence_flags_future_date(self):
        issues = check_coherence({"filling.date": "2999-01-01"})
        self.assertIn("data_no_futuro", issues)

    def test_coherence_flags_repeated_name_across_roles(self):
        issues = check_coherence({"mother.name": "Ana Paula Ferreira", "father.name": "Ana Paula Ferreira"})
        self.assertIn("nome_repetido_entre_papeis", issues)

    def test_coherence_municipio_uf_inconsistente(self):
        issues = check_coherence({"birth.place.ibge": "3304557", "birth.place.uf": "SP"})  # 33 = RJ, not SP
        self.assertIn("municipio_uf_inconsistente", issues)

    def test_cer_basic(self):
        self.assertEqual(cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(cer("abd", "abc"), 1 / 3, places=3)
        self.assertEqual(cer("", "abc"), 1.0)
        self.assertIsNone(cer("abc", ""))

    def test_normalize_loose(self):
        self.assertEqual(normalize_loose("JOÃO  da Silva"), "joao da silva")

    def test_cpf_check_digits(self):
        self.assertTrue(cpf_check_digits_ok("111.444.777-35"))
        self.assertFalse(cpf_check_digits_ok("111.444.777-00"))

    def test_validate_value_cep_uf(self):
        self.assertIsNone(validate_value("cep", "69005-040"))
        self.assertIsNotNone(validate_value("cep", "123"))
        self.assertIsNone(validate_value("uf", "AM"))
        self.assertIsNotNone(validate_value("uf", "ZZ"))


if __name__ == "__main__":
    unittest.main()
