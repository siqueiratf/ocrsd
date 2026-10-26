from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pymupdf

from src.mapping_assist import analysis, ocr
from src.mapping_assist.analysis import FieldCandidate, PageAnalysis, analyze_document, page_count
from src.mapping_assist.geometry import contained, iou
from src.mapping_assist.matching import Match, match_candidates
from src.mapping_assist.proposal import accept_suggestion, build_proposal
from src.web.services import mapping_issues, validate_mapping

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


def layout_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    line = {"color": (0, 0, 0), "width": 0.6}

    def anchor(number: str, x: float, y: float, label: str):
        page.draw_rect(pymupdf.Rect(x, y, x + 12, y + 12), **line)
        page.insert_text((x + 4, y + 10), number, fontsize=8)
        page.insert_text((x + 16, y + 10), label, fontsize=9)

    def option(x: float, y: float, code: str, label: str):
        page.draw_rect(pymupdf.Rect(x, y, x + 9, y + 9), **line)
        page.insert_text((x - 6, y + 8), code, fontsize=8)
        page.insert_text((x + 13, y + 8), label, fontsize=9)

    page.draw_rect(pymupdf.Rect(60, 100, 560, 180), **line)
    page.draw_line((240, 100), (240, 180), **line)
    anchor("1", 62, 102, "Data do registro")
    anchor("2", 242, 102, "Número do livro")
    for x in range(78, 313, 18):
        page.draw_line((x, 180), (x, 174), **line)
    page.draw_rect(pymupdf.Rect(60, 200, 560, 280), **line)
    anchor("3", 62, 202, "Código do cartório")
    for x in [70 + 18 * i for i in range(7)] + [178 + 14 * i for i in range(1, 6)]:
        page.draw_line((x, 280), (x, 274), **line)
    page.draw_rect(pymupdf.Rect(60, 300, 560, 380), **line)
    anchor("4", 62, 302, "Tipo de acidente")
    anchor("5", 300, 302, "Acidente do trabalho")
    for x, code, label in ((80, "1", "Trânsito"), (160, "2", "Queda"), (320, "1", "Sim"), (400, "2", "Não")):
        option(x, 340, code, label)
    page.draw_line((60, 400), (560, 400), **line)
    page.draw_line((60, 400), (60, 470), **line)
    page.draw_line((250, 400), (250, 470), **line)
    anchor("6", 62, 402, "A morte ocorreu")
    anchor("7", 252, 402, "Recebeu assistência")
    for x, code, label in ((80, "1", "Parto"), (150, "2", "Aborto"), (215, "9", "Ign"), (270, "1", "Sim"), (330, "2", "Não")):
        option(x, 425, code, label)
    return doc.tobytes()


def boxes_pdf() -> tuple[bytes, list[tuple[int, ...]], list[tuple[int, ...]]]:
    image = np.full((3508, 2480), 255, np.uint8)
    cv2.rectangle(image, (150, 150), (2330, 3350), 0, 3)
    cells, squares = [], []
    for row, width in enumerate((2, 4, 6, 8)):
        y = 400 + 300 * row
        cv2.rectangle(image, (300, y), (1500, y + 120), 0, width)
        cells.append((300 + width, y + width, 1500 - width, y + 120 - width))
        for i in range(3):
            x = 1700 + 120 * i
            cv2.rectangle(image, (x, y + 40), (x + 30, y + 70), 0, width)
            squares.append((x, y + 40, x + 30, y + 70))
    doc = pymupdf.open()
    page = doc.new_page(width=595.2, height=841.92)
    page.insert_image(page.rect, stream=cv2.imencode(".png", image)[1].tobytes())
    return doc.tobytes(), cells, squares


def inside(analysis: PageAnalysis) -> bool:
    width, height = analysis.page_size_px
    boxes = [c.region for c in analysis.candidates] + [c.label_bbox for c in analysis.candidates if c.label_bbox]
    boxes += [e["bbox"] for c in analysis.candidates for e in c.elements]
    return all(0 <= b[0] <= b[2] < width and 0 <= b[1] <= b[3] < height for b in boxes)


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
            self.assertTrue(0 <= c.region[0] <= c.region[2] < 2480 and 0 <= c.region[1] <= c.region[3] < 3509)
            self.assertIsNone(c.confidence)
        self.assertTrue(inside(a))
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
        second = analyze_document(two, page=1)
        proposal = build_proposal(second, match_candidates(second, "dnv"), "dnv")
        self.assertEqual((second.page, second.candidates), (1, ()))
        self.assertEqual((proposal["provenance"]["page"], proposal["fields"]), (1, []))
        self.assertTrue(any("página 2" in warning for warning in proposal["provenance"]["warnings"]))
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


class LayoutRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze_document(layout_pdf())

    def field(self, number: str, kind: str) -> list[FieldCandidate]:
        return [c for c in self.analysis.candidates if c.number == number and c.kind == kind]

    def test_combs_split_at_cell_borders_and_pitch_changes(self):
        (left,), (right,) = self.field("1", "digit_sequence"), self.field("2", "digit_sequence")
        self.assertEqual((left.label, len(left.elements), right.label, len(right.elements)), ("Data do registro", 10, "Número do livro", 4))
        self.assertLess(left.elements[-1]["bbox"][2], 240 * S)
        self.assertGreater(right.elements[0]["bbox"][0], 240 * S)
        self.assertEqual(sorted(len(c.elements) for c in self.field("3", "digit_sequence")), [5, 6])

    def test_checkbox_groups_follow_their_own_anchor(self):
        groups = {number: [[e["label"] for e in c.elements] for c in self.field(number, "checkbox")] for number in "4567"}
        self.assertEqual(groups, {"4": [["Trânsito", "Queda"]], "5": [["Sim", "Não"]], "6": [["Parto", "Aborto", "Ign"]], "7": [["Sim", "Não"]]})
        self.assertTrue(inside(self.analysis))


class RasterQualityTest(unittest.TestCase):
    def test_high_dpi_raster_is_downscaled(self):
        source = synthetic_pdf()
        base, high = analyze_document(raster_pdf(source, 300), ocr=False), analyze_document(raster_pdf(source, 1200), ocr=False)
        self.assertEqual((high.page_size_px, high.dpi), (base.page_size_px, base.dpi))
        self.assertEqual([c.kind for c in high.candidates], [c.kind for c in base.candidates])
        for a, b in zip(base.candidates, high.candidates):
            self.assertTrue(near(a.region, b.region, 4), (a.region, b.region))
            self.assertEqual(len(a.elements), len(b.elements))
            self.assertTrue(all(near(e["bbox"], f["bbox"], 4) for e, f in zip(a.elements, b.elements)))
        self.assertLess(abs(high.unit - base.unit), 1)
        self.assertLess(high.timings["total"], max(3 * base.timings["total"], base.timings["total"] + 2))
        self.assertTrue(inside(high))

    def test_thick_borders_without_text_unit(self):
        pdf, cells, squares = boxes_pdf()
        with mock.patch.object(analysis._Analyzer, "_hole_unit", return_value=None):
            a = analyze_document(pdf, ocr=False)
        self.assertEqual(a.unit, 15)
        texts = [c.region for c in a.candidates if c.kind == "text"]
        for cell in cells:
            self.assertGreater(max((iou(cell, t) for t in texts), default=0), 0.9, cell)
        found = [e["bbox"] for c in a.candidates if c.kind == "checkbox" for e in c.elements]
        self.assertEqual(len(found), len(squares))
        for square in squares:
            self.assertTrue(any(contained(f, square) == 1 and iou(f, square) > 0.4 for f in found), square)

    def test_upscale_warning_only_when_ocr_runs(self):
        low = raster_pdf(synthetic_pdf(), 90)
        self.assertFalse(any("ampliação" in w for w in analyze_document(low, ocr=False).warnings))
        with mock.patch.object(ocr, "run_tesseract", side_effect=FileNotFoundError("tesseract")):
            self.assertFalse(any("ampliação" in w for w in analyze_document(low).warnings))
        with mock.patch.object(ocr, "run_tesseract", return_value=[]):
            self.assertTrue(any("ampliação 3×" in w for w in analyze_document(low).warnings))

    def test_later_ocr_failures_are_reported(self):
        def fake(image, psm=6, dpi=None, options=(), timeout=180):
            if dpi is None:
                raise subprocess.TimeoutExpired("tesseract", timeout)
            return []

        with mock.patch.object(ocr, "run_tesseract", side_effect=fake):
            a = analyze_document(raster_pdf(synthetic_pdf()))
        self.assertEqual(a.source, "tesseract")
        failures = [w for w in a.warnings if w.startswith("Falha no OCR")]
        self.assertEqual(len(failures), 2, a.warnings)
        self.assertTrue(any("números" in w for w in failures) and any("títulos" in w for w in failures))


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


class CustomProposalTest(unittest.TestCase):
    def test_official_death_sex_boxes_keep_order_when_ocr_misreads_labels(self):
        elements = ({"label": "", "code": None, "bbox": [1381, 720, 1404, 737]},
                    {"label": "M-Mase.", "code": None, "bbox": [1381, 744, 1404, 761]},
                    {"label": "i-ignorado", "code": None, "bbox": [1381, 770, 1404, 788]})
        candidate = FieldCandidate("c001", "Sexo", (1320, 705, 1405, 718), (1320, 705, 1510, 790),
                                   "checkbox", elements=elements, number="10")
        analysis = PageAnalysis(0, 300, (2550, 3300), 12, "tesseract", (candidate,))
        match = Match("c001", "deceased.sex", "matched", "rule", 1.0, (), "Sexo na DO")
        proposal = validate_mapping(build_proposal(analysis, [match], "do"))
        self.assertEqual([item["value"] for item in proposal["fields"][0]["elements"]], ["M", "F", "I"])

    def test_approved_definition_is_embedded_for_generation(self):
        spec = {"key": "custom.registro.volume", "label": "Volume", "description": "Volume do registro",
                "data_type": "text", "generator": {"type": "choices", "values": ["A", "B"]}, "status": "approved"}
        candidate = FieldCandidate("c001", "Volume", (100, 100, 300, 120), (100, 120, 500, 180), "text")
        analysis = PageAnalysis(0, 300, (2480, 3509), 12, "text_layer", (candidate,))
        match = Match("c001", spec["key"], "matched", "custom", 1.0, ((spec["key"], 1.0),), "Definição aprovada")
        proposal = validate_mapping(build_proposal(analysis, [match], "do", custom_definitions=[spec]))
        self.assertEqual(proposal["fields"][0]["semantic"], spec["key"])
        self.assertEqual(proposal["semantic_definitions"][spec["key"]]["generator"]["values"], ["A", "B"])
        self.assertEqual(proposal["fields"][0]["suggestion"]["alternatives"][0]["label"], "Volume")
        self.assertEqual(accept_suggestion(proposal["fields"][0], spec["key"], custom_definitions=[spec])["semantic"], spec["key"])

    def test_boolean_definition_uses_supported_field_type(self):
        spec = {"key": "custom.registro.conferido", "label": "Conferido", "description": "Confirmação",
                "data_type": "boolean", "generator": {"type": "choices", "values": [True, False]}, "status": "approved"}
        candidate = FieldCandidate("c001", "Conferido", (100, 100, 300, 120), (100, 120, 500, 180), "text")
        analysis = PageAnalysis(0, 300, (2480, 3509), 12, "text_layer", (candidate,))
        match = Match("c001", spec["key"], "matched", "custom", 1.0, (), "Definição aprovada")
        proposal = validate_mapping(build_proposal(analysis, [match], "do", custom_definitions=[spec]))
        self.assertEqual(proposal["fields"][0]["data_type"], "text")


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
        self.assertTrue(inside(a))
        combs = {c.number: len(c.elements) for c in a.candidates if c.kind == "digit_sequence" and c.number}
        self.assertEqual({key: combs.get(key) for key in ("2", "3", "8", "9", "16", "46", "54", "55")},
                         {"2": 8, "3": 15, "8": 8, "9": 3, "16": 8, "46": 8, "54": 6, "55": 8})
        self.assertIn(combs.get("36"), {10, 11})
        self.assertLessEqual(combs.get("35", 0), 5)
        boxes = {c.number: c for c in a.candidates if c.kind == "checkbox" and c.number in {"37", "38", "48", "49"}}
        self.assertEqual((len(boxes["37"].elements), len(boxes["38"].elements)), (7, 3))
        self.assertTrue(all(e["bbox"][0] < 1285 for e in boxes["37"].elements) and all(e["bbox"][0] > 1285 for e in boxes["38"].elements))
        self.assertTrue(all(e["bbox"][0] < 1220 for e in boxes["48"].elements) and all(e["bbox"][0] > 1220 for e in boxes["49"].elements))
        self.assertTrue(any(c.label == "Hora" and c.kind == "text" for c in a.candidates))
        doctor = by_label(a, "Nome do Médico", "text")
        self.assertGreater(doctor.region[2], 1150)
        self.check_isolated_checkbox_proposals(a, "do", 3)

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
        self.assertTrue(inside(a))
        self.assertFalse(any(c.kind == "digit_sequence" and (c.region[1] < 300 or len(c.elements) < 2) for c in a.candidates))
        self.assertLess(a.timings["total"], 30)
        self.check_isolated_checkbox_proposals(a, "dnv", 2)

    def check_isolated_checkbox_proposals(self, analysis: PageAnalysis, context: str, maximum: int):
        isolated = {candidate.id for candidate in analysis.candidates
                    if candidate.kind == "checkbox" and len(candidate.elements) == 1}
        self.assertLessEqual(len(isolated), maximum)
        proposal = validate_mapping(build_proposal(analysis, match_candidates(analysis, context), context))
        fields = {field["suggestion"]["candidate_id"]: field for field in proposal["fields"]}
        sex_key = "newborn.sex" if context == "dnv" else "deceased.sex"
        sex = [field for field in fields.values() if field.get("semantic") == sex_key]
        self.assertEqual(len(sex), 1)
        self.assertEqual({element["value"] for element in sex[0]["elements"] if element["value"] in {"M", "F"}}, {"M", "F"})
        for identifier in isolated:
            field = fields[identifier]
            self.assertEqual(field["data_type"], "text")
            self.assertNotIn("elements", field)
            self.assertNotIn("semantic", field)
            self.assertEqual(field["suggestion"]["match"], "ambiguous")
            self.assertTrue(field["suggestion"]["issues"])
        self.assertFalse(any("checkbox precisa" in issue for issue in mapping_issues(proposal)))

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
        self.assertTrue(inside(a))


if __name__ == "__main__":
    unittest.main()
