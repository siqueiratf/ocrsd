import tempfile
import unittest
from pathlib import Path

import fitz

from src.pdf import keep_first_page_in_place


class KeepFirstPageTests(unittest.TestCase):
    @staticmethod
    def _make_pdf(path: Path, pages: int) -> None:
        document = fitz.open()
        for index in range(pages):
            page = document.new_page()
            page.insert_text((72, 72), f"page {index + 1}")
        document.save(path)
        document.close()

    def test_rewrites_multi_page_pdf_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.pdf"
            self._make_pdf(path, 3)

            page_count, changed = keep_first_page_in_place(str(path))

            with fitz.open(path) as result:
                self.assertEqual(len(result), 1)
                self.assertIn("page 1", result[0].get_text())
            self.assertEqual(page_count, 3)
            self.assertTrue(changed)

    def test_does_not_rewrite_single_page_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.pdf"
            self._make_pdf(path, 1)
            before = path.read_bytes()

            page_count, changed = keep_first_page_in_place(str(path))

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(page_count, 1)
            self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
