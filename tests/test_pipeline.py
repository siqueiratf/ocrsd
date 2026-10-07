import unittest

from src.pipeline import chunk_entries, parse_args


class ChunkEntriesTests(unittest.TestCase):
    def test_assigns_home_chunks_and_neighbor_overlap(self):
        entries = [{"word_id": index} for index in range(5)]

        chunks = chunk_entries(entries, chunk_size=2, overlap=1, id_field="word_id")

        self.assertEqual([entry["chunk_id"] for entry in entries], [0, 0, 1, 1, 2])
        self.assertEqual(
            chunks,
            [
                {"chunk_id": 0, "word_ids": [0, 1, 2]},
                {"chunk_id": 1, "word_ids": [1, 2, 3, 4]},
                {"chunk_id": 2, "word_ids": [3, 4]},
            ],
        )

    def test_empty_input_is_valid(self):
        self.assertEqual(chunk_entries([], 10, 2, "field_id"), [])

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_entries([], 0, 0, "word_id")
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_entries([], 1, -1, "word_id")


class CliTests(unittest.TestCase):
    def test_parse_args_accepts_pipeline_controls(self):
        args = parse_args(
            [
                "--document", "filled.pdf",
                "--template", "empty.pdf",
                "--output", "result",
                "--field-map", "fields.json",
                "--skip-ocr",
                "--stop-after", "diff",
            ]
        )

        self.assertEqual(args.document, "filled.pdf")
        self.assertEqual(args.field_map, "fields.json")
        self.assertTrue(args.skip_ocr)
        self.assertEqual(args.stop_after, "diff")


if __name__ == "__main__":
    unittest.main()
