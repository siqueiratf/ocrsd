import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.generated_pdf_viewer_gui import (
    GeneratedPdfViewerFrame, collect_pdfs, find_recent_dataset_folders, manifest_folder_for_pdf,
    pdf_display_name,
)


class GeneratedPdfViewerDiscoveryTests(unittest.TestCase):
    def test_finds_three_most_recent_dataset_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = []
            for index in range(5):
                folder = root / "documents" / "dnv" / "dataset" / f"20260915_120{index}-dnv-{index + 1}"
                folder.mkdir(parents=True)
                (folder / "dataset.json").write_text(json.dumps({"count": index}), encoding="utf-8")
                (folder / f"synthetic_{index:04d}.pdf").write_bytes(b"%PDF-1.7\n")
                timestamp = 1_700_000_000 + index
                for item in (folder, folder / "dataset.json", folder / f"synthetic_{index:04d}.pdf"):
                    os.utime(item, (timestamp, timestamp))
                folders.append(folder.resolve())

            with patch("src.generated_pdf_viewer_gui.discover_document_types", return_value={}):
                recent = find_recent_dataset_folders(3, root=root)

        self.assertEqual(len(recent), 3)
        self.assertEqual(
            [item.name for item in recent],
            ["20260915_1204-dnv-5", "20260915_1203-dnv-4", "20260915_1202-dnv-3"],
        )


class GeneratedPdfViewerBehaviorTests(unittest.TestCase):
    def test_collects_pdfs_from_selected_folder_and_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            top_pdf = root / "top.pdf"
            nested_pdf = child / "nested.pdf"
            top_pdf.write_bytes(b"%PDF-1.7\n")
            nested_pdf.write_bytes(b"%PDF-1.7\n")
            os.utime(top_pdf, (1_700_000_000, 1_700_000_000))
            os.utime(nested_pdf, (1_700_000_001, 1_700_000_001))

            pdfs = collect_pdfs(root)

        self.assertEqual(pdfs, [nested_pdf, top_pdf])
        self.assertEqual(pdf_display_name(top_pdf, root), "top.pdf")
        self.assertEqual(pdf_display_name(nested_pdf, root), "child/nested.pdf")

    def test_manifest_folder_prefers_nearest_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            (root / "dataset.json").write_text("{}", encoding="utf-8")
            (child / "dataset.json").write_text("{}", encoding="utf-8")
            pdf = child / "nested.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")

            folder = manifest_folder_for_pdf(pdf, root)

        self.assertEqual(folder, child)

    def test_zoom_changes_are_bounded_and_manual(self):
        frame = SimpleNamespace(zoom=1.0, fit_mode="page")
        frame._render = Mock()

        GeneratedPdfViewerFrame.change_zoom(frame, 20)
        self.assertEqual(frame.fit_mode, "manual")
        self.assertEqual(frame.zoom, 5.0)
        GeneratedPdfViewerFrame.change_zoom(frame, 0.001)
        self.assertEqual(frame.zoom, 0.15)
        self.assertEqual(frame._render.call_count, 2)

    def test_folder_list_is_trimmed_to_three_manual_open_targets(self):
        folders = [Path(f"/tmp/run_{index}") for index in range(5)]
        frame = SimpleNamespace(folders=folders[1:])
        frame.open_generated_folder = lambda folder: GeneratedPdfViewerFrame.open_generated_folder(frame, folder)
        frame._set_folders = Mock()
        with patch("src.generated_pdf_viewer_gui.filedialog.askdirectory", return_value=str(folders[0])):
            GeneratedPdfViewerFrame.open_folder(frame)

        frame._set_folders.assert_called_once_with(folders[:3])

    def test_open_generated_folder_loads_folder_first(self):
        folders = [Path("/tmp/run_1"), Path("/tmp/run_2")]
        frame = SimpleNamespace(folders=folders)
        frame._set_folders = Mock()

        GeneratedPdfViewerFrame.open_generated_folder(frame, Path("/tmp/run_3"))

        frame._set_folders.assert_called_once_with([Path("/tmp/run_3"), *folders])

    def test_manifest_is_updated_when_pdf_is_renamed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = folder / "dataset.json"
            manifest.write_text(json.dumps({
                "count": 1,
                "documents": [{"pdf": "old.pdf", "ground_truth": "old.json"}],
            }), encoding="utf-8")

            GeneratedPdfViewerFrame._replace_manifest_document(
                SimpleNamespace(), folder, "old.pdf", "new.pdf", "new.json",
            )

            data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["documents"], [{"pdf": "new.pdf", "ground_truth": "new.json"}])

    def test_manifest_is_updated_when_pdf_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = folder / "dataset.json"
            manifest.write_text(json.dumps({
                "count": 2,
                "documents": [
                    {"pdf": "keep.pdf", "ground_truth": "keep.json"},
                    {"pdf": "drop.pdf", "ground_truth": "drop.json"},
                ],
            }), encoding="utf-8")

            GeneratedPdfViewerFrame._remove_manifest_document(SimpleNamespace(), folder, "drop.pdf")

            data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["documents"], [{"pdf": "keep.pdf", "ground_truth": "keep.json"}])


if __name__ == "__main__":
    unittest.main()
