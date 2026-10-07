import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gui import collect_folder_images
from src.ocr_bench import OcrPrediction
from src.pipeline_gui import PipelineRunnerFrame


class FolderImageTests(unittest.TestCase):
    def test_collects_only_images_from_selected_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "field_0002.png").touch()
            (root / "field_0001.jpg").touch()
            (root / "notes.json").touch()
            debug = root / "debug"
            debug.mkdir()
            (debug / "aligned.png").touch()

            images = collect_folder_images(root)

        self.assertEqual([path.name for path in images], ["field_0001.jpg", "field_0002.png"])


class _SequentialBackend:
    def __init__(self):
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        return OcrPrediction(str(self.calls), 0.9, 2.0)


class DigitSequenceOcrTests(unittest.TestCase):
    def test_reads_each_element_in_index_order(self):
        field = {
            "field_id": "number", "data_type": "digit_sequence",
            "elements": [
                {"element_id": "second", "index": 1, "bbox": [20, 0, 30, 10]},
                {"element_id": "first", "index": 0, "bbox": [0, 0, 10, 10]},
            ],
        }
        backend = _SequentialBackend()
        with tempfile.TemporaryDirectory() as directory:
            predictions = PipelineRunnerFrame._recognize_digit_elements(
                np.zeros((20, 40, 3), dtype=np.uint8), field,
                {"fake": backend}, ["fake"], None, Path(directory),
            )

        self.assertEqual(backend.calls, 2)
        self.assertEqual(predictions["fake"]["text"], "12")
        by_id = {element["element_id"]: element for element in field["elements"]}
        self.assertEqual(by_id["first"]["ocr_text"], "1")
        self.assertEqual(by_id["second"]["ocr_text"], "2")

    def test_reads_text_sequence_in_index_order(self):
        field = {
            "field_id": "letters", "data_type": "text_sequence",
            "elements": [
                {"element_id": "second", "index": 1, "bbox": [20, 0, 30, 10]},
                {"element_id": "first", "index": 0, "bbox": [0, 0, 10, 10]},
            ],
        }
        backend = _SequentialBackend()
        with tempfile.TemporaryDirectory() as directory:
            predictions = PipelineRunnerFrame._recognize_sequence_elements(
                np.zeros((20, 40, 3), dtype=np.uint8), field,
                {"fake": backend}, ["fake"], None, Path(directory),
            )

        self.assertEqual(predictions["fake"]["text"], "12")


if __name__ == "__main__":
    unittest.main()
