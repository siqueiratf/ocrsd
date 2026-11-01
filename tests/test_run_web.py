import os
import subprocess
import unittest
from unittest.mock import patch

from scripts import run_web


class AiTunnelTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"OCRSD_AI_SSH_OPTIONS": "-o PubkeyAuthentication=no"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OCRSD_AI_BASE_URL", None)

    def test_tunnel_points_the_ai_url_to_the_local_port(self):
        with patch.object(run_web, "ensure_port"), patch.object(run_web.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            tunnel = run_web.open_ai_tunnel("ia@example")
            run_web.close_ai_tunnel(tunnel)
        command = run.call_args_list[0].args[0]
        self.assertIn("127.0.0.1:11435:127.0.0.1:11434", command)
        self.assertIn("PubkeyAuthentication=no", command)
        self.assertEqual(command[-1], "ia@example")
        self.assertEqual(os.environ["OCRSD_AI_BASE_URL"], "http://127.0.0.1:11435/v1")
        self.assertEqual(run.call_args_list[1].args[0][-3:], ["-O", "exit", "ia@example"])

    def test_failed_tunnel_stops_the_start(self):
        with patch.object(run_web, "ensure_port"), patch.object(run_web.subprocess, "run", return_value=subprocess.CompletedProcess([], 255)):
            with self.assertRaises(SystemExit):
                run_web.open_ai_tunnel("ia@example")
        self.assertNotIn("OCRSD_AI_BASE_URL", os.environ)


if __name__ == "__main__":
    unittest.main()
