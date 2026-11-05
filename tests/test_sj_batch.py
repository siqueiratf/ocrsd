import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from src.neural_handwriting import NeuralPen
from src.sj_handwriting import ROOT, SjHandwritingModel
from src.synthetic_dataset import _SampleRecorder, _SampleReplay, generate_document

PROJECT = Path(__file__).resolve().parents[1]
DNV_TEMPLATE = PROJECT / "documents/dnv/template.pdf"
DNV_MAPPING = PROJECT / "documents/dnv/mappings/mapeamento_seguindo_norma.json"
REQUESTS = [("Maria", 1, 0.75, 7), ("Florianopolis", 5, 0.3, 11), ("0", 9, 1.5, 3),
            ("Hospital Geral", 12, 0.75, 21), ("8", 1, 0.75, 5)]


def _close(test, first, second):
    test.assertEqual(first.shape, second.shape)
    test.assertLess(float(np.abs(first - second).max()), 1e-2)


@unittest.skipUnless((ROOT / "weights.sha256").is_file(), "Download sjvasquez weights for integration tests")
class SjBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model = SjHandwritingModel()

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_batch_is_close_to_single_rows_with_mixed_writers_and_biases(self):
        batch = self.model.sample_many(REQUESTS)
        for request, points in zip(REQUESTS, batch):
            _close(self, points, self.model.sample(*request))
        np.testing.assert_array_equal(self.model.sample(*REQUESTS[0]), self.model.sample_many(REQUESTS[:1])[0])

    def test_rows_do_not_depend_on_the_other_rows(self):
        alone = self.model.sample_many(REQUESTS[:1])[0]
        mixed = self.model.sample_many([REQUESTS[3], REQUESTS[1], REQUESTS[0]])[2]
        _close(self, alone, mixed)
        previous = self.model._max_batch
        try:
            self.model._max_batch = 2
            queued = self.model.sample_many(REQUESTS)
        finally:
            self.model._max_batch = previous
        for first, second in zip(queued, self.model.sample_many(REQUESTS)):
            _close(self, first, second)

    def test_batch_is_deterministic(self):
        for first, second in zip(self.model.sample_many(REQUESTS), self.model.sample_many(REQUESTS)):
            np.testing.assert_array_equal(first, second)

    def test_error_rows_do_not_stop_the_other_rows(self):
        requests = [("Maria@", 1, 0.75, 1), REQUESTS[0], ("Maria", 13, 0.75, 1), ("x" * 76, 1, 0.75, 1)]
        results = self.model.sample_many(requests)
        self.assertEqual([isinstance(result, ValueError) for result in results], [True, False, True, True])
        np.testing.assert_array_equal(results[1], self.model.sample(*REQUESTS[0]))
        self.assertEqual(self.model.sample_many([]), [])
        with self.assertRaises(ValueError):
            self.model.sample("Maria", 13, 0.75, 1)

    def test_row_that_does_not_finish_gets_an_error(self):
        row = self.model._row

        def short(text, writer, bias, seed):
            values = row(text, writer, bias, seed)
            if text == "Hospital Geral":
                values["limit"] = 3
            return values

        with patch.object(self.model, "_row", side_effect=short):
            results = self.model.sample_many(REQUESTS)
        self.assertIsInstance(results[3], ValueError)
        self.assertIn("did not finish", str(results[3]))
        _close(self, results[0], self.model.sample(*REQUESTS[0]))


class _SeedModel:
    """Stand-in model: the stroke depends on the text and the seed. Some seeds and texts fail."""

    def __init__(self, fail_every=0, fail_text=None, fail_seeds=()):
        self.fail_every, self.fail_text, self.fail_seeds, self.calls = fail_every, fail_text, set(fail_seeds), 0

    def check(self, text, writer, bias):
        if text == self.fail_text:
            raise ValueError("stub refuses the text")
        return text

    def sample(self, text, writer, bias, seed):
        self.calls += 1
        self.check(text, writer, bias)
        if (self.fail_every and seed % self.fail_every == 0) or seed in self.fail_seeds:
            raise ValueError("stub failure")
        rng = np.random.default_rng(seed)
        points = [[0.0, 0.0, 1.0]]
        for index, _ in enumerate(text):
            for step in range(4):
                points.append([index * 12 + step * 3, float(rng.normal(0, 3)), float(step == 3)])
        return np.array(points, dtype=np.float32)


class _BatchSeedModel(_SeedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []

    def sample_many(self, requests):
        self.batches.append(list(requests))
        results = []
        for request in requests:
            try:
                results.append(_SeedModel.sample(self, *request))
            except ValueError as exc:
                results.append(exc)
        self.calls -= len(requests)
        return results


class RecordReplayTests(unittest.TestCase):
    def _document(self, model, directory, seed=4242):
        field_map = json.loads(DNV_MAPPING.read_text(encoding="utf-8"))
        selected = {str(field["field_id"]) for field in field_map["fields"] if not field.get("unsupported_reason")}
        pen = NeuralPen(model="sjvasquez", writer=None, text_policy="normalize")
        document = generate_document(DNV_TEMPLATE, field_map, selected, Path(directory), "doc", random.Random(seed),
                                     neural_pen=pen, handwriting_model={"sjvasquez": model})
        truth = json.loads(document.json_path.read_text(encoding="utf-8"))
        with fitz.open(document.pdf_path) as pdf:
            pixels = pdf[0].get_pixmap(dpi=60).samples
        return truth, pixels

    def _assert_same_document(self, fail_every=0, fail_text=None, fail_seeds=(), seed=4242):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            single = _SeedModel(fail_every, fail_text, fail_seeds)
            batch = _BatchSeedModel(fail_every, fail_text, fail_seeds)
            expected = self._document(single, first, seed)
            actual = self._document(batch, second, seed)
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertGreater(single.calls, 100)
        return batch

    def test_replay_gives_the_same_document_and_hits_every_request(self):
        batch = self._assert_same_document()
        self.assertEqual(len(batch.batches), 1)
        self.assertEqual(batch.calls, 0)

    def test_failed_requests_make_a_new_planning_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _BatchSeedModel()
            self._document(first, directory)
        seeds = [request[3] for request in first.batches[0]]
        batch = self._assert_same_document(fail_seeds=(seeds[20], seeds[100]))
        self.assertEqual(len(batch.batches), 3)
        self.assertEqual(batch.calls, 0)

    def test_text_that_always_fails_falls_back_to_the_font(self):
        batch = self._assert_same_document(fail_text="Florianopolis")
        self.assertEqual(len(batch.batches), 1)
        self.assertEqual(batch.calls, 12)

    def test_many_failures_stay_correct_with_misses(self):
        self._assert_same_document(fail_every=5)

    def test_recorder_and_replay(self):
        recorder = _SampleRecorder({("b", 1, 0.5, 2): ValueError("failed")})
        self.assertEqual(recorder.sample("a", 1, 0.5, 1).shape[1], 3)
        with self.assertRaises(ValueError):
            recorder.sample("b", 1, 0.5, 2)
        self.assertEqual(recorder.requests, [("a", 1, 0.5, 1)])
        model = _SeedModel()
        points = model.sample("a", 1, 0.5, 1)
        replay = _SampleReplay(model, {("a", 1, 0.5, 1): points, ("b", 1, 0.5, 2): ValueError("failed")})
        returned = replay.sample("a", 1, 0.5, 1)
        np.testing.assert_array_equal(returned, points)
        returned[:] = 0
        np.testing.assert_array_equal(replay.sample("a", 1, 0.5, 1), points)
        with self.assertRaises(ValueError):
            replay.sample("b", 1, 0.5, 2)
        replay.sample("c", 1, 0.5, 3)
        self.assertEqual((replay.hits, replay.misses), (3, 1))


if __name__ == "__main__":
    unittest.main()
