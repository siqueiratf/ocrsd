import unittest
from unittest.mock import Mock, patch

import numpy as np

from PIL import Image

from src.handwriting_gallery_gui import GalleryVariation, SAMPLE_TEXTS, render_gallery_samples


class HandwritingGalleryTests(unittest.TestCase):
    def test_font_gallery_renders_three_variations_for_each_sample(self):
        rows = render_gallery_samples(
            "font", None, 0.75, "normalize",
            [
                GalleryVariation("#193787", 1.4, 0.2, 0.0),
                GalleryVariation("#1f1f24", 2.1, 0.4, 0.05),
                GalleryVariation("#315f9d", 1.0, 0.7, 0.1),
            ],
        )

        self.assertEqual(len(rows), len(SAMPLE_TEXTS))
        for row in rows:
            self.assertEqual(len(row["images"]), 3)
            self.assertTrue(all(isinstance(image, Image.Image) for image in row["images"]))

    def test_wpvnteam_gallery_uses_neural_model(self):
        model = Mock()
        model.sample.return_value = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 1]], dtype=float)
        with patch("src.handwriting_gallery_gui.load_model", return_value=model):
            rows = render_gallery_samples(
                "wpvnteam", 3, 2.1, "normalize",
                [GalleryVariation("#193787", 1.4, 0.2, 0.0)],
            )

        self.assertEqual(len(rows), len(SAMPLE_TEXTS))
        self.assertEqual(model.sample.call_count, len(SAMPLE_TEXTS))


if __name__ == "__main__":
    unittest.main()
