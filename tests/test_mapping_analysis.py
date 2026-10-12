from __future__ import annotations

import difflib
import json
import shutil
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pymupdf

from src.mapping_assist import ocr
from src.mapping_assist.analysis import FieldCandidate, PageAnalysis, analyze_document, page_count
from src.mapping_assist.geometry import iou

ROOT = Path(__file__).resolve().parents[1]
S = 300 / 72
TESSERACT = shutil.which("tesseract")
NO_TESSERACT = "tesseract não está instalado; testes de OCR ignorados"
TICKS = [70 + 18 * i for i in range(9)]
SQUARES = [(340, 185, 349, 194), (440, 185, 449, 194)]


def px(box) -> tuple[float, ...]:
    return tuple(v * S for v in box)


def near(a, b, tolerance: float = 3) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def similar(labels, query: str) -> float:
    return max((difflib.SequenceMatcher(None, norm(query), norm(label)).ratio() for label in labels), default=0.0)


def synthetic_pdf(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    line = {"color": (0, 0, 0), "width": 0.6}
    for top, bottom, title in ((100, 400, "Identificação"), (420, 700, "Endereço")):
        page.draw_rect(pymupdf.Rect(30, top, 48, bottom), color=None, fill=(0, 0, 0))
        page.insert_text((43, bottom - 20), title, fontsize=9, rotate=90, color=(1, 1, 1))
    page.draw_rect(pymupdf.Rect(60, 100, 560, 230), **line)
    page.draw_line((60, 160), (560, 160), **line)
    page.draw_line((330, 160), (330, 230), **line)
    for number, (x, y), label in (("1", (62, 102), "Nome completo"), ("2", (62, 162), "Data de nascimento"), ("3", (332, 162), "Sexo")):
        page.draw_rect(pymupdf.Rect(x, y, x + 12, y + 12), **line)
        page.insert_text((x + 4, y + 10), number, fontsize=8)
        page.insert_text((x + 16, y + 10), label, fontsize=9)
    for x in TICKS:
        page.draw_line((x, 230), (x, 224), **line)
    for (x1, y1, x2, y2), code, option in zip(SQUARES, ("1", "2"), ("Masculino", "Feminino")):
        page.draw_rect(pymupdf.Rect(x1, y1, x2, y2), **line)
        page.insert_text((x1 - 6, y2 - 1), code, fontsize=8)
        page.insert_text((x2 + 4, y2 - 1), option, fontsize=9)
    page.insert_text((60, 441), "Município de residência", fontsize=6)
    page.draw_rect(pymupdf.Rect(60, 444, 400, 466), **line)
    page.insert_text((420, 441), "UF", fontsize=6)
    page.draw_rect(pymupdf.Rect(420, 444, 470, 466), **line)
    for _ in range(pages - 1):
        doc.new_page(width=595, height=842)
    return doc.tobytes()


def raster_pdf(source: bytes, dpi: int = 300) -> bytes:
    pix = pymupdf.open(stream=source, filetype="pdf")[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    return doc.tobytes()


def by_label(analysis: PageAnalysis, query: str, kind: str | None = None) -> FieldCandidate:
    pool = [c for c in analysis.candidates if kind is None or c.kind == kind]
    return max(pool, key=lambda c: difflib.SequenceMatcher(None, norm(query), norm(c.label)).ratio())


class SyntheticTextLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = synthetic_pdf()
        cls.analysis = analyze_document(cls.pdf)
        cls.words = pymupdf.open(stream=cls.pdf, filetype="pdf")[0].get_text("words")

    def word_box(self, top: float, *texts: str) -> tuple[float, ...]:
        boxes = [w[:4] for w in self.words if w[4] in texts and abs(w[3] - top) < 5]
        return px((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)))

    def test_contract(self):
        a = self.analysis
        self.assertEqual((a.page, a.dpi, a.page_size_px, a.source), (0, 300, (2480, 3509), "text_layer"))
        self.assertGreater(a.unit, 10)
        self.assertEqual([c.id for c in a.candidates], [f"c{i:03d}" for i in range(1, len(a.candidates) + 1)])
        for c in a.candidates:
            self.assertIn(c.kind, {"text", "digit_sequence", "checkbox"})
            self.assertTrue(all(isinstance(v, int) for v in c.region))
            self.assertTrue(0 <= c.region[0] <= c.region[2] <= 2480 and 0 <= c.region[1] <= c.region[3] <= 3509)
            self.assertIsNone(c.confidence)
        self.assertIn("total", a.timings)

    def test_text_field(self):
        c = by_label(self.analysis, "Nome completo")
        self.assertEqual((c.label, c.kind, c.number, c.section, c.source), ("Nome completo", "text", "1", "I", "text_layer"))
        self.assertTrue(near(c.label_bbox, self.word_box(112, "Nome", "completo")))
        cell = px((60, 100, 560, 160))
        self.assertTrue(c.region[1] >= c.label_bbox[3] and c.region[3] <= cell[3] and c.region[0] >= cell[0] - 3 and c.region[2] <= cell[2] + 3)
        self.assertGreater(c.region[2] - c.region[0], 0.9 * (cell[2] - cell[0]))

    def test_digit_sequence(self):
        c = by_label(self.analysis, "Data de nascimento", "digit_sequence")
        self.assertEqual((c.label, c.number, c.section), ("Data de nascimento", "2", "I"))
        self.assertEqual([e["index"] for e in c.elements], list(range(8)))
        for e, (a, b) in zip(c.elements, zip(TICKS, TICKS[1:])):
            x1, y1, x2, y2 = e["bbox"]
            self.assertTrue(near((x1, x2), (a * S + 6, b * S - 6)), e)
            self.assertTrue(abs(y2 - 230 * S) <= 3 and abs((y2 - y1) - 0.8 * 18 * S) <= 3, e)

    def test_checkbox(self):
        c = by_label(self.analysis, "Sexo", "checkbox")
        self.assertEqual((c.label, c.number, c.section), ("Sexo", "3", "I"))
        self.assertEqual([(e["label"], e["code"]) for e in c.elements], [("Masculino", "1"), ("Feminino", "2")])
        for e, square in zip(c.elements, SQUARES):
            self.assertTrue(near(e["bbox"], px(square)), e)
        self.assertTrue(all(c.region[0] <= e["bbox"][0] and e["bbox"][2] <= c.region[2] for e in c.elements))

    def test_boxes_below_labels(self):
        for label, box in (("Município de residência", (60, 444, 400, 466)), ("UF", (420, 444, 470, 466))):
            c = by_label(self.analysis, label, "text")
            self.assertEqual((c.label, c.section), (label, "II"))
            self.assertGreater(iou(c.region, px(box)), 0.85)
            self.assertTrue(near(c.label_bbox, self.word_box(441, *label.split())))

    def test_sections_and_neighbors(self):
        self.assertEqual({c.section for c in self.analysis.candidates}, {"I", "II"})
        self.assertEqual(by_label(self.analysis, "UF").section_title, "Endereço")
        self.assertIn("Nome completo", by_label(self.analysis, "Data de nascimento").neighbors)
        self.assertIn("Data de nascimento", by_label(self.analysis, "Sexo").neighbors)

    def test_page_helpers(self):
        self.assertEqual(page_count(self.pdf), 1)
        two = synthetic_pdf(pages=2)
        self.assertEqual(page_count(two), 2)
        self.assertTrue(any("2 páginas" in w for w in analyze_document(two).warnings))
        with self.assertRaises(ValueError):
            analyze_document(two, page=5)


class RasterFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector = analyze_document(synthetic_pdf())
        cls.pdf = raster_pdf(synthetic_pdf())

    @unittest.skipIf(TESSERACT is None, NO_TESSERACT)
    def test_tesseract_labels_and_geometry(self):
        a = analyze_document(self.pdf)
        self.assertEqual(a.source, "tesseract")
        labels = [c.label for c in a.candidates]
        for query in ("Nome completo", "Data de nascimento", "Sexo", "Município de residência"):
            self.assertGreaterEqual(similar(labels, query), 0.8, query)
        digits = by_label(a, "Data de nascimento", "digit_sequence")
        self.assertEqual(len(digits.elements), 8)
        expected = by_label(self.vector, "Data de nascimento", "digit_sequence")
        for e, f in zip(digits.elements, expected.elements):
            self.assertTrue(near(e["bbox"], f["bbox"], 6), (e, f))
        box = by_label(a, "Sexo", "checkbox")
        self.assertEqual([e["code"] for e in box.elements], ["1", "2"])
        self.assertGreaterEqual(similar([e["label"] for e in box.elements], "Feminino"), 0.8)
        for e, square in zip(box.elements, SQUARES):
            self.assertTrue(near(e["bbox"], px(square), 4), e)
        self.assertTrue(all(0 <= (c.confidence or 0) <= 1 for c in a.candidates))
        self.assertEqual(by_label(a, "Sexo", "checkbox").number, "3")

    def test_missing_tesseract_degrades_to_geometry(self):
        with mock.patch.object(ocr, "run_tesseract", side_effect=FileNotFoundError("tesseract")):
            a = analyze_document(self.pdf)
        self.assertEqual(a.source, "geometry")
        self.assertTrue(any("Tesseract não encontrado" in w for w in a.warnings))
        kinds = {c.kind: c for c in a.candidates if c.kind != "text"}
        self.assertEqual(len(kinds["digit_sequence"].elements), 8)
        self.assertEqual(len(kinds["checkbox"].elements), 2)
        self.assertTrue(all(c.label == "" and c.source == "geometry" for c in a.candidates))

    def test_ocr_disabled(self):
        a = analyze_document(self.pdf, ocr=False)
        self.assertEqual(a.source, "geometry")
        self.assertTrue(a.warnings)


class PlacementTest(unittest.TestCase):
    def test_offset_anisotropic_image(self):
        image = np.full((1500, 1600), 255, np.uint8)
        cv2.rectangle(image, (100, 100), (1500, 1400), 0, 4)
        squares = [(300 + 220 * i, 600, 340 + 220 * i, 640) for i in range(4)]
        for x1, y1, x2, y2 in squares:
            cv2.rectangle(image, (x1, y1), (x2, y2), 0, 3)
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        rect = pymupdf.Rect(50, 60, 350, 320)
        page.insert_image(rect, stream=cv2.imencode(".png", image)[1].tobytes(), keep_proportion=False)
        a = analyze_document(doc.tobytes(), ocr=False)
        self.assertEqual(a.page_size_px, (2550, 3300))
        sx, sy = rect.width * S / 1600, rect.height * S / 1500
        found = [e["bbox"] for c in a.candidates if c.kind == "checkbox" for e in c.elements]
        self.assertEqual(len(found), 4)
        for x1, y1, x2, y2 in squares:
            expected = (rect.x0 * S + (x1 + 1.5) * sx, rect.y0 * S + (y1 + 1.5) * sy, rect.x0 * S + (x2 - 1.5) * sx, rect.y0 * S + (y2 - 1.5) * sy)
            self.assertTrue(any(near(box, expected) for box in found), (expected, found))


@unittest.skipIf(TESSERACT is None, NO_TESSERACT)
class RealDocumentsTest(unittest.TestCase):
    def test_declaracao_de_obito(self):
        a = analyze_document(ROOT / "documents/do/do.pdf")
        self.assertEqual((a.source, a.page_size_px), ("tesseract", (2550, 3300)))
        self.assertGreaterEqual(sum(len(c.elements) for c in a.candidates if c.kind == "checkbox"), 60)
        self.assertGreaterEqual(len({c.number for c in a.candidates if c.number and c.label}), 50)
        labels = [c.label for c in a.candidates]
        for query in ("Nome do Falecido", "Data do óbito", "Município de ocorrência", "CRM"):
            self.assertGreaterEqual(similar(labels, query), 0.85, query)
        self.assertEqual({c.section for c in a.candidates if c.section}, {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"})
        self.assertLess(a.timings["total"], 30)

    def test_declaracao_de_nascido_vivo(self):
        a = analyze_document(ROOT / "documents/dnv/template.pdf")
        mapping = json.loads((ROOT / "documents/dnv/mappings/mapeamento_seguindo_norma.json").read_text())
        self.assertEqual(a.page_size_px, tuple(mapping["page_size_px"]))
        squares = [e["bbox"] for c in a.candidates if c.kind == "checkbox" for e in c.elements]
        manual = [e["bbox"] for f in mapping["fields"] if f.get("data_type") == "checkbox" for e in f["elements"]]
        self.assertGreaterEqual(sum(max(iou(m, s) for s in squares) >= 0.5 for m in manual), 0.9 * len(manual))
        digits = [e["bbox"] for c in a.candidates if c.kind == "digit_sequence" for e in c.elements]
        for f in mapping["fields"]:
            if f["field_id"] in {"2", "10", "15_cpf", "24", "50"}:
                hits = sum(max(iou(e["bbox"], d) for d in digits) >= 0.3 for e in f["elements"])
                self.assertGreaterEqual(hits, 0.8 * len(f["elements"]), f["field_id"])
        self.assertTrue(any("baixa resolução" in w for w in a.warnings))
        self.assertLess(a.timings["total"], 30)

    def test_certidao_text_layer(self):
        a = analyze_document(ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf")
        mapping = json.loads((ROOT / "documents/certidao_nascimento/certidao_nascimento_fields.json").read_text())
        self.assertEqual((a.source, a.page_size_px), ("text_layer", tuple(mapping["page_size_px"])))
        regions = [c.region for c in a.candidates]
        self.assertGreaterEqual(sum(max(iou(f["bbox"], r) for r in regions) >= 0.3 for f in mapping["fields"]), 13)
        labels = [c.label for c in a.candidates]
        for query in ("Data de nascimento", "Horário de nascimento", "Município da naturalidade", "Sexo", "DNV"):
            self.assertIn(query, labels)
        self.assertLess(a.timings["total"], 30)


if __name__ == "__main__":
    unittest.main()
