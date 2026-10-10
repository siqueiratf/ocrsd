from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from zipfile import ZipFile

try:
    from fastapi.testclient import TestClient
    import jsonschema
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.jobs import JobStore
from src.web.models import GenerationOptions
from src.web.services import Catalog, generate
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0, neural_enabled=False)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.store = self.app.state.store
        self.options = GenerationOptions(document_type="certidao_nascimento", mapping_id="certidao_nascimento_fields",
                                         fields=["nome_registrado"], count=1, seed=42)
        definition = Catalog(self.settings).definition(self.options.document_type, self.options.mapping_id)
        self.options.fields = [field["field_id"] for field in definition["fields"][:3]]

    def submit(self):
        response = self.client.post("/api/v1/jobs", json=self.options.model_dump())
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        return job, {"Authorization": "Bearer " + job["token"]}

    def worker_environment(self):
        return {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()},
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_worker(self):
        return subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT,
                              env=self.worker_environment(), capture_output=True, text=True, timeout=90, check=True)

    def assert_contract(self, response, path, method="get"):
        schema = self.app.openapi()
        definition = schema["paths"][path][method]["responses"][str(response.status_code)]["content"]["application/json"]["schema"]
        jsonschema.Draft202012Validator({**definition, "components": schema["components"]}).validate(response.json())

    def test_catalog_contract_and_no_tkinter_import(self):
        for path in ("health", "document-types", "capabilities", "writing-presets"):
            response = self.client.get("/api/v1/" + path)
            self.assertEqual(response.status_code, 200)
            self.assert_contract(response, "/api/v1/" + path)
        for document in self.client.get("/api/v1/document-types").json():
            template = self.client.get(f'/api/v1/document-types/{document["id"]}/template')
            self.assertTrue(template.content.startswith(b"%PDF"))
            for mapping in document["mappings"]:
                response = self.client.get(f'/api/v1/document-types/{document["id"]}/mappings/{mapping["id"]}')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["definition"]["source_template"], document["id"])
                self.assert_contract(response, "/api/v1/document-types/{document_type}/mappings/{mapping_id}")
        result = subprocess.run([sys.executable, "-c", "import sys; from src.web.api import create_app; assert 'tkinter' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_strict_validation_and_bounded_body(self):
        changes = [{"count": 0}, {"count": True}, {"count": "1"}, {"count": 11}, {"seed": -1},
                   {"fields": []}, {"fields": ["unknown"]}, {"fields": self.options.fields * 2},
                   {"document_type": "../dnv"}, {"output": "/tmp/escape"}, {"filter_preset": "other"},
                   {"writing": {"model": "sjvasquez"}}, {"writing": {"writer": 512}},
                   {"writing": {"model": "sjvasquez", "writer": 13}}, {"writing": {"intensity": "NaN"}}]
        for update in changes:
            with self.subTest(update=update):
                response = self.client.post("/api/v1/generation/validate", json={**self.options.model_dump(), **update})
                self.assertEqual(response.status_code, 422, response.text)
                self.assert_contract(response, "/api/v1/generation/validate", "post")
        response = self.client.post("/api/v1/jobs", content="{}", headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 415)
        response = self.client.post("/api/v1/jobs", content="x" * 32769, headers={"Content-Type": "application/json", "Content-Length": "2"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.get("/api/v1/missing").json()["error"]["code"], "http_error")

    def test_real_worker_download_preview_and_determinism(self):
        import fitz

        outputs = []
        for iteration in range(2):
            job, headers = self.submit()
            self.run_worker()
            base = f'/api/v1/jobs/{job["id"]}'
            response = self.client.get(base, headers=headers)
            self.assertEqual(response.json()["status"], "succeeded", response.text)
            self.assertEqual(response.json()["completed"], 1)
            self.assert_contract(response, "/api/v1/jobs/{job_id}")
            response = self.client.get(base + "/results", headers=headers)
            self.assert_contract(response, "/api/v1/jobs/{job_id}/results")
            files = response.json()["artifacts"]
            pdf = next(item for item in files if item["media_type"] == "application/pdf")
            preview = self.client.get(base + f'/artifacts/{pdf["id"]}?download=false', headers=headers)
            self.assertIn("inline", preview.headers["content-disposition"])
            with fitz.open(stream=preview.content, filetype="pdf") as document:
                self.assertIn("SEM VALIDADE LEGAL", document[0].get_text())
                raster = document[0].get_pixmap().samples
            directory = self.store.directory(job["id"])
            ground_truth = json.loads((directory / "synthetic_0001.json").read_text())
            self.assertTrue(ground_truth["synthetic"])
            self.assertNotIn(str(ROOT), json.dumps(ground_truth))
            with ZipFile(directory / "dataset.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"synthetic_0001.pdf", "synthetic_0001.json", "dataset.json"})
                self.assertTrue(json.loads(archive.read("dataset.json"))["synthetic"])
            outputs.append((ground_truth, raster))
        self.assertEqual(outputs[0], outputs[1])

    def test_artifact_isolation_cancel_and_expiration(self):
        first, headers = self.submit()
        second, second_headers = self.submit()
        base = f'/api/v1/jobs/{first["id"]}'
        self.assertEqual(self.client.get(base).status_code, 401)
        self.assertEqual(self.client.get(base, headers=second_headers).status_code, 404)
        self.assertEqual(self.client.get(base + "/results", headers=headers).status_code, 409)
        response = self.client.post(base + "/cancel", headers=headers, json={})
        self.assertEqual(response.json()["status"], "cancelled")
        self.assert_contract(response, "/api/v1/jobs/{job_id}/cancel", "post")
        directory = self.store.directory(second["id"])
        directory.mkdir()
        outside = self.settings.storage / "private.txt"
        outside.write_text("privado")
        (directory / "link.txt").symlink_to(outside)
        self.store.finish(second["id"], "succeeded", artifacts=[{"id": "0", "name": "link.txt", "media_type": "text/plain", "size": 7}])
        url = f'/api/v1/jobs/{second["id"]}/artifacts/0'
        self.assertEqual(self.client.get(url, headers=second_headers).status_code, 404)
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET expires_at=? WHERE id=?", (time.time() - 1, second["id"]))
        self.assertEqual(self.client.get(url, headers=second_headers).status_code, 410)
        self.store.cleanup()
        self.assertFalse(directory.exists())
        self.assertTrue(outside.exists())

    def test_concurrent_requests_respect_atomic_queue_limit(self):
        self.settings.max_pending = 2
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(lambda _: self.client.post("/api/v1/jobs", json=self.options.model_dump()), range(8)))
        self.assertEqual([response.status_code for response in responses].count(202), 2)
        self.assertEqual([response.status_code for response in responses].count(429), 6)
        self.assertEqual(len({response.json()["id"] for response in responses if response.status_code == 202}), 2)
        with ThreadPoolExecutor(max_workers=4) as executor:
            claimed = list(executor.map(lambda _: self.store.claim(), range(4)))
        self.assertEqual(len([row for row in claimed if row]), 2)

    def test_rate_storage_and_maintenance_limits(self):
        self.settings.jobs_per_hour = 1
        self.submit()
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).json()["error"]["code"], "rate_limit")
        self.settings.accepting_jobs = False
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).status_code, 503)
        self.settings.accepting_jobs = True
        self.settings.jobs_per_hour = 10
        self.settings.max_storage_bytes = self.settings.max_job_bytes
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).json()["error"]["code"], "storage_quota")

    def test_worker_failure_and_recovery(self):
        job, headers = self.submit()
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET options='{}' WHERE id=?", (job["id"],))
        self.run_worker()
        response = self.client.get(f'/api/v1/jobs/{job["id"]}', headers=headers)
        self.assertEqual(response.json()["status"], "failed")
        self.assertNotIn("Traceback", response.text)
        self.assertFalse(self.store.directory(job["id"]).exists())
        queued, token = self.store.submit(self.options, "another-client")
        self.store.claim()
        self.store.recover()
        self.assertEqual(self.store.get(queued["id"])["status"], "failed")

    def test_running_cancellation_and_timeout(self):
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                self.settings.max_count = 100
                self.settings.job_timeout = 90 if cancel else 1
                self.options.count = 100
                job, headers = self.submit()
                process = subprocess.Popen([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT,
                                           env=self.worker_environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    deadline = time.monotonic() + 15
                    while self.store.get(job["id"])["status"] == "queued" and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual(self.store.get(job["id"])["status"], "running")
                    if cancel:
                        self.client.post(f'/api/v1/jobs/{job["id"]}/cancel', headers=headers, json={})
                    process.wait(timeout=20)
                    self.assertEqual(self.store.get(job["id"])["status"], "cancelled" if cancel else "failed")
                    self.assertFalse(self.store.directory(job["id"]).exists())
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()

    def test_service_reuses_dnv_engine_and_filters(self):
        options = GenerationOptions(document_type="dnv", mapping_id="mapeamento_seguindo_norma", fields=["1"], filter_preset="light")
        definition = Catalog(self.settings).definition(options.document_type, options.mapping_id)
        options.fields = [definition["fields"][0]["field_id"]]
        progress = []
        output = self.settings.storage / "service"
        artifacts = generate(options, output, self.settings, progress.append)
        self.assertEqual(progress, [1])
        self.assertEqual(len(artifacts), 4)
        data = json.loads((output / "synthetic_0001.json").read_text())
        self.assertTrue(data["visual_filters"]["enabled"])
        self.assertEqual(data["fields"][0]["field_id"], options.fields[0])
        with self.assertRaises(FileExistsError):
            generate(options, output, self.settings, progress.append)

    def test_postman_matches_openapi_and_examples(self):
        from scripts.export_postman import build_collection

        collection = json.loads((ROOT / "postman/ocrsd.postman_collection.json").read_text())
        self.assertEqual(collection, build_collection())
        operations = set()
        schema = self.app.openapi()
        for folder in collection["item"]:
            for item in folder["item"]:
                request = item["request"]
                if isinstance(request["url"], str):
                    continue
                path = request["url"]["raw"].removeprefix("{{baseUrl}}").split("?")[0].replace("{{", "{").replace("}}", "}").replace("{pdf_id}", "{artifact_id}")
                method = request["method"].lower()
                operations.add((path, method))
                for response in item["response"]:
                    if response["header"][0]["value"] == "application/json":
                        definition = schema["paths"][path][method]["responses"][str(response["code"])]["content"]["application/json"]["schema"]
                        jsonschema.Draft202012Validator({**definition, "components": schema["components"]}).validate(json.loads(response["body"]))
        self.assertEqual(operations, {(path, method) for path, methods in schema["paths"].items() for method in methods})


if __name__ == "__main__":
    unittest.main()
