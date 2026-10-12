import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.mapping_assist.providers import NullAssistant, OpenAICompatibleAssistant, assistant_from_settings
from src.web.settings import Settings


class ModelListHandler(BaseHTTPRequestHandler):
    payload = {}

    def log_message(self, *_args):
        pass

    def do_GET(self):
        body = json.dumps(type(self).payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProviderAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ModelListHandler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def test_disabled_provider_does_not_contact_server(self):
        self.assertIsInstance(assistant_from_settings(Settings(ai_provider="none"), None), NullAssistant)

    def test_requires_model_inventory_and_exact_model(self):
        provider = OpenAICompatibleAssistant(self.url, "qwen3:1.7b")
        for payload in ({}, {"data": None}, {"data": []}, {"data": [{"id": "other"}]},
                        {"data": [{"id": None}]}):
            ModelListHandler.payload = payload
            self.assertFalse(provider.status().available, payload)
        ModelListHandler.payload = {"data": [{"id": "qwen3:1.7b"}]}
        self.assertTrue(provider.status().available)


if __name__ == "__main__":
    unittest.main()
