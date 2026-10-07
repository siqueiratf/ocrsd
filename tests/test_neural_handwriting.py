import json
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import fitz
import numpy as np

from src.neural_handwriting import (
    ALPHABET, WEIGHTS_PATH, HandwritingModel, NeuralPen, prepare_text, render_strokes,
)
from src.sj_handwriting import ROOT, SjHandwritingModel
from src.synthetic_dataset import generate_dataset


class PenTests(unittest.TestCase):
    def test_alphabet_and_explicit_normalization(self):
        self.assertEqual(len(ALPHABET), 73)
        with self.assertRaises(ValueError):
            prepare_text("João / XZQ")
        self.assertEqual(prepare_text("João / XZQ", "normalize"), "Joao - xzq")
        with self.assertRaises(ValueError):
            prepare_text("teste@", "normalize")

    def test_model_specific_style_limits(self):
        NeuralPen(model="aaron1259", writer=511)
        NeuralPen(model="wpvnteam", writer=511)
        NeuralPen(model="sjvasquez", writer=12)
        NeuralPen(width=40)
        for kwargs in ({"writer": 13}, {"width": 0}, {"width": 40.1}, {"color": "blue"}, {"dropout": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NeuralPen(**kwargs)

    def test_wpvnteam_metadata_records_source(self):
        metadata = NeuralPen(model="wpvnteam", writer=7).metadata()

        self.assertEqual(metadata["repository"], "wpvnteam/handwriting")
        self.assertEqual(metadata["model"], "wpvnteam")

    def test_pen_controls_and_repeatability(self):
        x = np.arange(120, dtype=float)
        points = np.column_stack((x, np.sin(x / 8), np.zeros(120)))
        pen = NeuralPen(color="#126334", width=2, pressure=0)
        clean = np.asarray(render_strokes(points, 800, 90, pen, 42))
        repeated = np.asarray(render_strokes(points, 800, 90, pen, 42))
        thick = np.asarray(render_strokes(points, 800, 90, replace(pen, width=5), 42))
        broken = np.asarray(render_strokes(points, 800, 90, replace(pen, dropout=1), 42))
        np.testing.assert_array_equal(clean, repeated)
        self.assertGreater(thick[:, :, 3].sum(), clean[:, :, 3].sum())
        self.assertLess(broken[:, :, 3].sum(), clean[:, :, 3].sum())
        self.assertTrue((clean[:, :, :3] == (18, 99, 52)).all())
        self.assertFalse(clean[0, :, 3].any())
        self.assertFalse(clean[-1, :, 3].any())

    def test_pen_up_does_not_connect_separate_strokes(self):
        points = np.array([[0, 0, 0], [10, 0, 1], [90, 0, 0], [100, 0, 1]], dtype=float)
        image = np.asarray(render_strokes(points, 200, 40, NeuralPen(pressure=0), 42))
        self.assertFalse(image[:, 80:120, 3].any())

    def test_bad_weights_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.bin"
            path.write_bytes(b"not the model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                HandwritingModel(path)


class NeuralDatasetTests(unittest.TestCase):
    def test_checkbox_uses_selected_pen_color(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, mapping = root / "template.pdf", root / "mapping.json"
            with fitz.open() as document:
                document.new_page(width=100, height=100)
                document.save(template)
            mapping.write_text(json.dumps({"dpi": 72, "template_id": "test", "fields": [
                {"field_id": "choice", "label": "Choice", "data_type": "checkbox", "bbox": [10, 10, 40, 40],
                 "elements": [{"element_id": "yes", "value": "Y", "bbox": [10, 10, 40, 40]}]},
            ]}))
            model = Mock()
            model.sample.return_value = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 1]], dtype=float)
            with patch("src.synthetic_dataset.load_model", return_value=model):
                result = generate_dataset(template, mapping, {"choice"}, root / "out", 1, seed=42,
                                          neural_pen=NeuralPen(color="#ff0000", pressure=0, width=2))
            model.sample.assert_called_once()
            with fitz.open(result[0].pdf_path) as document:
                pixmap = document[0].get_pixmap()
                rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(100, 100, 3)
            self.assertTrue(((rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 120) & (rgb[:, :, 2] < 120)).any())
            truth = json.loads(result[0].json_path.read_text())
            rendering = truth["fields"][0]["elements"][0]["rendering"]
            self.assertEqual(rendering["backend"], "sjvasquez")
            self.assertIn(rendering["mark_variant"], {"middle_stroke", "upper_stroke", "vertical_stroke", "x"})

    def test_models_keep_same_values_and_record_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, mapping = root / "template.pdf", root / "mapping.json"
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping.write_text(json.dumps({"dpi": 300, "template_id": "test", "fields": [
                {"field_id": "name", "label": "Nome", "bbox": [100, 100, 600, 200],
                 "validation": {"allowed_values": ["João da Silva", "José Pereira"]}},
            ]}))
            model = Mock()
            model.sample.return_value = np.array([[0, 0, 0], [10, 1, 0], [20, 0, 1]], dtype=float)
            truths = []
            for backend in ("sjvasquez", "aaron1259"):
                with patch("src.synthetic_dataset.load_model", return_value=model) as load:
                    result = generate_dataset(template, mapping, {"name"}, root / backend, 2, seed=42,
                                              neural_pen=NeuralPen(model=backend, text_policy="normalize"))
                    load.assert_called_once_with(backend)
                truth = [json.loads(item.json_path.read_text()) for item in result]
                truths.append(truth)
                for item in truth:
                    self.assertEqual(item["handwriting"]["model"], backend)
                    field = item["fields"][0]
                    self.assertEqual(field["value"], prepare_text(field["requested_value"], "normalize"))
                    self.assertEqual(field["rendering"]["backend"], backend)
            self.assertEqual([item["fields"][0]["value"] for item in truths[0]],
                             [item["fields"][0]["value"] for item in truths[1]])

    def test_can_use_different_models_for_numbers_and_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, mapping = root / "template.pdf", root / "mapping.json"
            with fitz.open() as document:
                document.new_page()
                document.save(template)
            mapping.write_text(json.dumps({"dpi": 300, "template_id": "test", "fields": [
                {"field_id": "digits", "label": "Digits", "data_type": "digit_sequence",
                 "bbox": [100, 100, 220, 160],
                 "elements": [{"index": 0, "bbox": [100, 100, 150, 160]}]},
                {"field_id": "name", "label": "Nome", "bbox": [100, 200, 600, 260],
                 "validation": {"allowed_values": ["Ana Silva"]}},
            ]}))
            model = Mock()
            model.sample.return_value = np.array([[0, 0, 0], [10, 1, 0], [20, 0, 1]], dtype=float)

            with patch("src.synthetic_dataset.load_model", return_value=model):
                result = generate_dataset(
                    template, mapping, {"digits", "name"}, root / "out", 1, seed=42,
                    numeric_neural_pen=NeuralPen(model="sjvasquez", writer=1, text_policy="normalize"),
                    text_neural_pen=NeuralPen(model="aaron1259", writer=2, text_policy="normalize"),
                )

            truth = json.loads(result[0].json_path.read_text())
            by_id = {field["field_id"]: field for field in truth["fields"]}
            self.assertEqual(by_id["digits"]["elements"][0]["rendering"]["backend"], "sjvasquez")
            self.assertEqual(by_id["name"]["rendering"]["backend"], "aaron1259")
            self.assertEqual(truth["handwriting"]["numbers"]["model"], "sjvasquez")
            self.assertEqual(truth["handwriting"]["text"]["model"], "aaron1259")


class LocalModelTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "weights.sha256").is_file() and importlib.util.find_spec("tensorflow"),
                         "Requires local checkpoint and TensorFlow conversion extra")
    def test_converted_lstm_matches_tensorflow_cell(self):
        import tensorflow as tf
        import torch

        model = SjHandwritingModel()
        reader = tf.train.load_checkpoint(str(ROOT / "checkpoints/model-17900"))
        rng = np.random.default_rng(7)
        for index in range(3):
            prefix = "rnn/LSTMAttentionCell/lstm_cell" + (f"_{index}" if index else "")
            kernel = reader.get_tensor(prefix + "/kernel")
            bias = reader.get_tensor(prefix + "/bias")
            np.testing.assert_array_equal(kernel, model.weights[prefix + "/kernel"].numpy())
            inputs = rng.normal(0, 0.1, (1, kernel.shape[0]-400)).astype("float32")
            h = rng.normal(0, 0.1, (1, 400)).astype("float32")
            c = rng.normal(0, 0.1, (1, 400)).astype("float32")


            gates = np.r_[0:400, 800:1200, 400:800, 1200:1600]
            keras_bias = bias[gates].copy()
            keras_bias[400:800] += 1
            cell = tf.keras.layers.LSTMCell(400, unit_forget_bias=False)
            cell.build((1, inputs.shape[1]))
            cell.set_weights([kernel[:-400, gates], kernel[-400:, gates], keras_bias])
            _, expected = cell(tf.constant(inputs), [tf.constant(h), tf.constant(c)])
            actual = model._lstm(torch.from_numpy(inputs), (torch.from_numpy(h), torch.from_numpy(c)), index)
            for value, reference in zip(actual, expected):
                np.testing.assert_allclose(value.numpy(), reference.numpy(), atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(WEIGHTS_PATH.is_file(), "Download aaron1259 weights for integration tests")
    def test_aaron_model_is_deterministic_and_visible(self):
        import torch
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = HandwritingModel()
            first = model.sample("Maria", 0, 2.1, 42)
            np.testing.assert_array_equal(first, model.sample("Maria", 0, 2.1, 42))
            self.assertGreater(np.asarray(render_strokes(first, 400, 100, NeuralPen(model="aaron1259"), 42))[:, :, 3].sum(), 1000)
        finally:
            torch.set_num_threads(previous)

    @unittest.skipUnless((ROOT / "weights.sha256").is_file(), "Download sjvasquez weights for integration tests")
    def test_sj_model_is_deterministic_and_visible(self):
        import torch
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = SjHandwritingModel()
            first = model.sample("Maria", 9, 0.75, 42)
            np.testing.assert_array_equal(first, model.sample("Maria", 9, 0.75, 42))
            self.assertGreater(np.asarray(render_strokes(first, 400, 100, NeuralPen(), 42))[:, :, 3].sum(), 1000)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
