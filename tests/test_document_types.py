import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.document_types import (
    build_dataset_run_dir, collect_dataset_pdfs, discover_document_types,
    infer_document_type_key,
)


class DocumentTypeTests(unittest.TestCase):
    def test_discovers_complete_document_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dnv = root / "dnv"
            (dnv / "mappings").mkdir(parents=True)
            (dnv / "presets").mkdir()
            (dnv / "dataset").mkdir()
            (dnv / "template.pdf").touch()
            (dnv / "mappings" / "b.json").touch()
            (dnv / "mappings" / "a.json").touch()

            result = discover_document_types(root)

        self.assertEqual(list(result), ["dnv"])
        self.assertEqual([path.name for path in result["dnv"].mappings], ["a.json", "b.json"])

    def test_ignores_folder_without_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "incomplete").mkdir()
            self.assertEqual(discover_document_types(root), {})

    def test_ignores_folder_without_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "sem_mapa"
            document.mkdir()
            (document / "sem_mapa.pdf").touch()

            self.assertEqual(discover_document_types(root), {})

    def test_discovers_birth_certificate_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certificate = root / "certidao_nascimento"
            certificate.mkdir()
            (certificate / "certidao_nascimento.pdf").touch()
            (certificate / "certidao_nascimento_fields.json").touch()

            result = discover_document_types(root)

        self.assertIn("certidao_nascimento", result)
        self.assertEqual(result["certidao_nascimento"].template.name, "certidao_nascimento.pdf")
        self.assertEqual(
            [path.name for path in result["certidao_nascimento"].mappings],
            ["certidao_nascimento_fields.json"],
        )

    def test_collects_only_direct_pdf_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.PDF").touch()
            (root / "a.pdf").touch()
            (root / "image.png").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "hidden.pdf").touch()

            result = collect_dataset_pdfs(root)

        self.assertEqual([path.name for path in result], ["a.pdf", "b.PDF"])

    def test_builds_dated_dataset_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_dataset_run_dir("dnv", 12, root=root, now=datetime(2026, 9, 15, 8, 7))

        self.assertEqual(path, Path(directory) / "dnv" / "dataset" / "20260915_0807-dnv-12")

    def test_infers_document_type_from_documents_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "certidao_nascimento" / "template.pdf"
            mapping = root / "certidao_nascimento" / "mappings" / "map.json"

            self.assertEqual(infer_document_type_key(template, mapping, root=root), "certidao_nascimento")


if __name__ == "__main__":
    unittest.main()
