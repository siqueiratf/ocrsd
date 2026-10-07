import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts.publish_release import main, release_notes


class ReleaseTests(unittest.TestCase):
    def test_notes_include_only_requested_version(self):
        with tempfile.TemporaryDirectory() as directory:
            changelog = Path(directory) / "CHANGELOG.md"
            changelog.write_text("# Changelog\n\n## [V2] — hoje\n\nNovo.\n\n## [V1] — antes\n\nAntigo.\n")
            self.assertEqual(release_notes("V2", changelog), "Novo.\n")
            with self.assertRaisesRegex(ValueError, "V3"):
                release_notes("V3", changelog)

    def run_publication(self, root: Path, responses: list):
        changelog = root / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [V1] — hoje\n\nVersão validada.\n")
        asset = root / "source.tar.gz"
        asset.write_bytes(b"synthetic source archive")
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token", "GITHUB_REPOSITORY": "owner/repo"}), \
             patch("scripts.publish_release.request_json", side_effect=responses) as request, \
             patch.object(Path, "write_text") as write, contextlib.redirect_stdout(io.StringIO()):
            main(["--tag", "V1", "--changelog", str(changelog), "--assets", str(asset)])
        return request, write

    def test_publishes_draft_after_asset_upload(self):
        draft = {"id": 7, "draft": True, "upload_url": "https://uploads.github.com/test{?name}", "assets": []}
        asset = {"name": "source.tar.gz", "size": 24, "browser_download_url": "https://github.com/test/source.tar.gz"}
        published = {"id": 7, "html_url": "https://github.com/test/V1", "published_at": "2026-10-07"}
        missing = urllib.error.HTTPError("https://api.github.com/test", 404, "missing", {}, None)
        with tempfile.TemporaryDirectory() as directory:
            request, write = self.run_publication(Path(directory), [missing, draft, asset, published])
        calls = request.call_args_list
        self.assertTrue(calls[1].kwargs["payload"]["draft"])
        self.assertEqual(calls[2].kwargs["data"], b"synthetic source archive")
        self.assertFalse(calls[3].kwargs["payload"]["draft"])
        report = json.loads(write.call_args.args[0])
        self.assertEqual(report["assets"][0]["name"], "source.tar.gz")

    def test_failed_upload_does_not_publish_draft(self):
        draft = {"id": 7, "draft": True, "upload_url": "https://uploads.github.com/test{?name}", "assets": []}
        missing = urllib.error.HTTPError("https://api.github.com/test", 404, "missing", {}, None)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(OSError, "upload failed"):
            self.run_publication(Path(directory), [missing, draft, OSError("upload failed")])


if __name__ == "__main__":
    unittest.main()
