import unittest

import numpy as np

from src.preprocess import adjust_tone, blue_ink_mask, preprocess_for_ocr


class PreprocessTests(unittest.TestCase):
    def test_blue_ink_mask_excludes_black_form_lines(self):
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        image[4:8, 4:8] = (180, 80, 20)
        image[12:16, 12:16] = (0, 0, 0)

        mask = blue_ink_mask(image, min_component_area=1, close_kernel=1)

        self.assertEqual(int(mask[5, 5]), 255)
        self.assertEqual(int(mask[13, 13]), 0)

    def test_adjust_tone_clips_to_uint8(self):
        image = np.array([[0, 100, 250]], dtype=np.uint8)
        result = adjust_tone(image, brightness=20)
        np.testing.assert_array_equal(result, np.array([[20, 120, 255]], dtype=np.uint8))

    def test_rejects_unknown_mode(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "desconhecido"):
            preprocess_for_ocr(image, mode="invalid")


if __name__ == "__main__":
    unittest.main()
