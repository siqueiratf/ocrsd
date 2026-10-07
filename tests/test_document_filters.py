import random
import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np

from src.document_filters import apply_document_filters, apply_realistic_filters_to_pdf


class DocumentFilterTests(unittest.TestCase):
    def test_filters_are_deterministic_and_preserve_shape(self):
        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[20:60, 25:95] = (210, 220, 235)

        first, first_metadata = apply_document_filters(image, random.Random(123), "medium")
        second, second_metadata = apply_document_filters(image, random.Random(123), "medium")

        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_metadata, second_metadata)
        self.assertTrue(first_metadata["enabled"])
        self.assertEqual(first_metadata["preset"], "medium")
        self.assertFalse(np.array_equal(first, image))

    def test_applies_filters_to_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            target = root / "target.pdf"
            with fitz.open() as document:
                page = document.new_page(width=120, height=160)
                page.insert_text((20, 40), "Teste")
                document.save(source)

            metadata = apply_realistic_filters_to_pdf(source, target, dpi=72, rng=random.Random(4), preset="light")

            self.assertTrue(target.is_file())
            self.assertEqual(metadata["preset"], "light")
            with fitz.open(target) as document:
                self.assertEqual(len(document), 1)


if __name__ == "__main__":
    unittest.main()
