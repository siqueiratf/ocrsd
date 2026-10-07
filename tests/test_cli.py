import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import fitz
import numpy as np

from src.config import PipelineConfig
from src import ocr, ocr_bench
from src.cli import main


class HeadlessCliTests(unittest.TestCase):
    def test_run_passes_engine_and_dpi_to_pipeline(self):
        with patch("src.pipeline.run_pipeline") as run:
            main(["run", "--document", "filled.pdf", "--template", "empty.pdf",
                  "--output", "output", "--ocr-engine", "tesseract", "--dpi", "150"])
        config = run.call_args.args[3]
        self.assertEqual((config.ocr_engine, config.dpi), ("tesseract", 150))

    def test_preprocess_preserves_subfolders_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "nested" / "image.png"
            source.parent.mkdir(parents=True)
            cv2.imwrite(str(source), np.full((30, 60, 3), 255, dtype=np.uint8))
            original = source.read_bytes()
            preset = root / "preset.json"
            preset.write_text(json.dumps({"preset_version": 1, "mode": "traditional",
                                          "parameters": {"grayscale_only": True}}))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["preprocess", "--input", str(root / "input"), "--output", str(root / "out"),
                      "--preset", str(preset)])
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((root / "out" / "nested" / "image.png").is_file())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["preprocess", "--input", str(root / "input"), "--output", str(root / "input" / "out")])
            self.assertEqual(error.exception.code, 1)

    def test_generate_outputs_pdf_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({"template_id": "test", "dpi": 300, "fields": [
                {"field_id": "uf", "label": "UF", "data_type": "uf", "bbox": [100, 100, 300, 200]},
            ]}))
            output = root / "documents" / "dnv" / "dataset" / "20260915_0807-dnv-1"
            with patch("src.synthetic_dataset.build_dataset_run_dir", return_value=output), contextlib.redirect_stdout(io.StringIO()):
                main(["generate", "--template", str(template), "--field-map", str(mapping),
                      "--count", "1", "--seed", "7", "--printed"])
            self.assertTrue((output / "synthetic_0001.pdf").is_file())
            truth = json.loads((output / "synthetic_0001.json").read_text())
            self.assertEqual(truth["fields"][0]["field_id"], "uf")

    def test_generate_accepts_wpvnteam_handwriting_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({"template_id": "test", "dpi": 300, "fields": [
                {"field_id": "name", "label": "Nome", "bbox": [100, 100, 300, 200]},
            ]}))
            with patch("src.synthetic_dataset.generate_dataset") as generate, contextlib.redirect_stdout(io.StringIO()):
                generate.return_value = []
                main(["generate", "--template", str(template), "--field-map", str(mapping),
                      "--count", "1", "--handwriting-model", "wpvnteam"])

            self.assertEqual(generate.call_args.kwargs["neural_pen"].model, "wpvnteam")

    def test_generate_passes_realistic_filter_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({"template_id": "test", "dpi": 300, "fields": [
                {"field_id": "name", "label": "Nome", "bbox": [100, 100, 300, 200]},
            ]}))
            with patch("src.synthetic_dataset.generate_dataset") as generate, contextlib.redirect_stdout(io.StringIO()):
                generate.return_value = []
                main(["generate", "--template", str(template), "--field-map", str(mapping),
                      "--count", "1", "--realistic-filters", "--filter-preset", "strong"])

            self.assertTrue(generate.call_args.kwargs["realistic_filters"])
            self.assertEqual(generate.call_args.kwargs["filter_preset"], "strong")
            self.assertIsNone(generate.call_args.args[3])

    def test_generate_output_is_optional_for_dated_document_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "documents" / "dnv" / "template.pdf"
            template.parent.mkdir(parents=True)
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping = root / "documents" / "dnv" / "mappings" / "map.json"
            mapping.parent.mkdir(parents=True)
            mapping.write_text(json.dumps({"template_id": "test", "dpi": 300, "fields": [
                {"field_id": "name", "label": "Nome", "bbox": [100, 100, 300, 200]},
            ]}))
            generated_doc = Mock()
            generated_doc.pdf_path = root / "documents" / "dnv" / "dataset" / "20260915_0807-dnv-1" / "synthetic_0001.pdf"
            with patch("src.synthetic_dataset.generate_dataset", return_value=[generated_doc]) as generate, contextlib.redirect_stdout(io.StringIO()):
                main(["generate", "--template", str(template), "--field-map", str(mapping), "--count", "1"])

            self.assertIsNone(generate.call_args.args[3])

    def test_benchmark_saves_preprocessed_single_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.png"
            cv2.imwrite(str(source), np.full((30, 60, 3), 255, dtype=np.uint8))
            backend = Mock()
            backend.recognize.return_value = ocr_bench.OcrPrediction("SP", 0.9, 1.0)
            with patch.dict(ocr_bench.BACKEND_FACTORIES, {"fake": lambda: backend}), contextlib.redirect_stdout(io.StringIO()):
                main(["benchmark", "--input", str(source), "--models", "fake",
                      "--preprocess", "traditional", "--save-preprocessed", str(root / "processed"),
                      "--output", str(root / "result.json")])
            self.assertTrue((root / "processed" / "input.png").is_file())

    def test_unavailable_backend_returns_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.png"
            cv2.imwrite(str(source), np.full((30, 60, 3), 255, dtype=np.uint8))
            factory = Mock(side_effect=RuntimeError("unavailable"))
            with patch.dict(ocr_bench.BACKEND_FACTORIES, {"fake": factory}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["benchmark", "--input", str(source), "--models", "fake", "--output", str(root / "result.json")])
            self.assertEqual(error.exception.code, 1)

    def test_alternative_pipeline_backend_is_cached(self):
        backend = Mock()
        backend.recognize.return_value = ocr_bench.OcrPrediction("SP", None, 1.0)
        factory = Mock(return_value=backend)
        with patch.dict(ocr_bench.BACKEND_FACTORIES, {"fake": factory}), patch.dict(ocr._backend_cache, {}, clear=True):
            image = np.zeros((10, 20, 3), dtype=np.uint8)
            config = PipelineConfig(ocr_engine="fake")
            self.assertEqual(ocr.recognize_word(image, config).text, "SP")
            self.assertEqual(ocr.recognize_line(image, config).confidence, 0.0)
            factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
